import os
import json
import logging
import time
import io
from multiprocessing import Pool, cpu_count
from functools import partial # 导入 partial
from PIL import Image, TiffTags # Corrected from TiffImagePlugin
import img2pdf
import fitz  # PyMuPDF

# --- 全局配置和日志记录器 ---
# CONFIG 不再是全局变量，而是通过参数传递给 worker 函数
logger = logging.getLogger(__name__)

# --- 1. 配置和日志设置 ---
def load_config(config_path='tif2pdf_config_v2.json'):
    """加载配置文件"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logger.error(f"配置文件 {config_path} 未找到。")
        raise
    except json.JSONDecodeError:
        logger.error(f"配置文件 {config_path} 格式错误。")
        raise

def setup_logging(logging_settings):
    """配置日志记录"""
    log_level = logging_settings.get("level", "INFO").upper()
    log_format = logging_settings.get("format", "%(asctime)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=log_level, format=log_format)

# --- 2. 核心转换逻辑 ---
def get_tiff_resolution_and_dimensions(tiff_path):
    """获取TIFF图像的分辨率和尺寸，优先使用TiffTags"""
    try:
        with Image.open(tiff_path) as img:
            # 尝试从TiffTags获取分辨率
            x_resolution = img.tag.get(TiffTags.RESOLUTION_UNIT) # XResolution (282)
            y_resolution = img.tag.get(TiffTags.Y_RESOLUTION) # YResolution (283)
            resolution_unit = img.tag.get(TiffTags.RESOLUTION_UNIT) # ResolutionUnit (296)

            if x_resolution and y_resolution and resolution_unit:
                # ResolutionUnit: 1 (无单位), 2 (英寸), 3 (厘米)
                if resolution_unit == 2: # 英寸
                    dpi = (x_resolution[0][0] / x_resolution[0][1], y_resolution[0][0] / y_resolution[0][1])
                elif resolution_unit == 3: # 厘米
                    dpi = ( (x_resolution[0][0] / x_resolution[0][1]) * 2.54, (y_resolution[0][0] / y_resolution[0][1]) * 2.54)
                else: # 无单位或未知，默认使用PIL的DPI属性
                    dpi = img.info.get('dpi', (72, 72)) # 默认72 DPI
            else:
                dpi = img.info.get('dpi', (72, 72)) # 默认72 DPI

            width, height = img.size
            return dpi, (width, height)
    except Exception as e:
        logger.warning(f"无法读取 {tiff_path} 的元数据，将使用默认值: {e}")
        return (72, 72), (0, 0) # 返回默认DPI和无效尺寸

def should_downsample(original_dpi, target_dpi, dimensions, max_dimension_pixels=8000):
    """判断是否需要下采样，基于DPI和尺寸"""
    # 如果原始DPI高于目标DPI，则需要下采样
    if original_dpi[0] > target_dpi or original_dpi[1] > target_dpi:
        return True
    # 如果图像尺寸过大（例如，超过8000像素），即使DPI不高，也可能需要下采样以避免内存问题
    if dimensions[0] > max_dimension_pixels or dimensions[1] > max_dimension_pixels:
        logger.info(f"图像尺寸 {dimensions} 较大，将进行下采样。")
        return True
    return False

def downsample_image(image, target_dpi):
    """将图像下采样到目标DPI"""
    try:
        original_dpi = image.info.get('dpi', (target_dpi, target_dpi))
        if original_dpi[0] == 0 or original_dpi[1] == 0 : # 处理DPI为0的情况
            original_dpi = (target_dpi, target_dpi)

        scale_factor_x = target_dpi / original_dpi[0]
        scale_factor_y = target_dpi / original_dpi[1]
        scale_factor = min(scale_factor_x, scale_factor_y) # 使用较小的比例因子以保持纵横比

        if scale_factor < 1: # 仅当需要缩小时才操作
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            logger.info(f"下采样图像从 {image.size} (DPI: {original_dpi}) 到 { (new_width, new_height)} (目标DPI: {target_dpi})")
            resized_image = image.resize((new_width, new_height), Image.LANCZOS) # 使用高质量的LANCZOS滤波器
            resized_image.info['dpi'] = (target_dpi, target_dpi)
            return resized_image
        return image # 如果不需要缩小，返回原图
    except Exception as e:
        logger.error(f"下采样图像失败: {e}")
        return image # 出错时返回原图

def convert_single_tiff_to_pdf_bytes(tiff_path, conversion_settings):
    """将单个TIFF文件（可能多页）转换为PDF字节流，进行优化处理"""
    try:
        target_dpi = conversion_settings.get("target_dpi", 150)
        jpeg_quality = conversion_settings.get("jpeg_quality", 85)
        downsample_enabled = conversion_settings.get("downsample_images", True)
        # color_grayscale_compression = conversion_settings.get("color_grayscale_compression", "JPEG").upper() # JPEG 或 ZIP/Deflate

        img_bytes_list = []
        with Image.open(tiff_path) as img:
            for i in range(img.n_frames):
                img.seek(i)
                current_image = img.copy() # 操作副本，避免影响原始图像对象

                # 检查并转换调色板图像 (P模式) 或 RGBA/LA (透明度)
                if current_image.mode == 'P':
                    logger.info(f"页面 {i+1} 在 {tiff_path} 是调色板图像，转换为RGB。")
                    current_image = current_image.convert('RGB')
                elif current_image.mode in ('RGBA', 'LA'):
                    logger.info(f"页面 {i+1} 在 {tiff_path} 包含Alpha通道，转换为RGB。")
                    # 创建一个白色背景，然后将图像粘贴到上面以去除透明度
                    background = Image.new('RGB', current_image.size, (255, 255, 255))
                    background.paste(current_image, mask=current_image.split()[3]) # 使用alpha通道作为蒙版
                    current_image = background


                original_dpi, dimensions = get_tiff_resolution_and_dimensions(tiff_path) # 获取原始DPI和尺寸
                # 确保原始DPI有效
                if original_dpi[0] == 0 or original_dpi[1] == 0:
                    logger.warning(f"TIFF文件 {tiff_path} 的原始DPI为0，将使用目标DPI {target_dpi} 作为原始DPI。")
                    original_dpi = (target_dpi, target_dpi)


                if downsample_enabled and should_downsample(original_dpi, target_dpi, dimensions):
                    current_image = downsample_image(current_image, target_dpi)
                else:
                    # 即使不下采样，也确保图像有DPI信息，img2pdf需要
                    if 'dpi' not in current_image.info:
                         current_image.info['dpi'] = original_dpi if original_dpi[0]!=0 and original_dpi[1]!=0 else (target_dpi,target_dpi)


                # 将处理后的图像帧保存为内存中的JPEG字节流 (或PNG如果需要无损)
                frame_bytes = io.BytesIO()
                if current_image.mode == '1': # 黑白图像 (1-bit pixels, black and white)
                    # 对于黑白图像，img2pdf可以直接处理，不需要转JPEG，以保持清晰度
                    # 但如果强制JPEG压缩，则需要转为灰度或RGB
                    # if color_grayscale_compression == "JPEG":
                    #    current_image = current_image.convert('L') # 转为灰度进行JPEG压缩
                    #    current_image.save(frame_bytes, format='JPEG', quality=jpeg_quality, dpi=current_image.info.get('dpi'))
                    # else: # 使用PNG (无损) 或等待img2pdf的CCITTFaxDecode
                    current_image.save(frame_bytes, format='PNG', dpi=current_image.info.get('dpi')) # PNG通常对黑白图像更有效
                elif current_image.mode == 'L': # 灰度图像
                    # if color_grayscale_compression == "JPEG":
                    current_image.save(frame_bytes, format='JPEG', quality=jpeg_quality, dpi=current_image.info.get('dpi'))
                    # else: # 使用PNG (无损)
                    #    current_image.save(frame_bytes, format='PNG', dpi=current_image.info.get('dpi'))
                else: # 彩色图像 (RGB等)
                    current_image.save(frame_bytes, format='JPEG', quality=jpeg_quality, dpi=current_image.info.get('dpi'), subsampling=0) # subsampling=0 保留色度信息

                img_bytes_list.append(frame_bytes.getvalue())

        if not img_bytes_list:
            logger.warning(f"没有从 {tiff_path} 提取到任何图像帧。")
            return None

        # 使用img2pdf将图像字节列表转换为PDF字节流
        pdf_bytes = img2pdf.convert(img_bytes_list)
        return pdf_bytes

    except Image.DecompressionBombError as e:
        logger.error(f"处理 {tiff_path} 时发生Pillow解压炸弹错误: {e}。请检查Image.MAX_IMAGE_PIXELS设置。")
        return None
    except Exception as e:
        logger.error(f"转换 {tiff_path} 失败: {e}", exc_info=True)
        return None

def optimize_pdf(pdf_bytes, output_pdf_path):
    """使用PyMuPDF优化PDF（重新组织、清除、线性化）"""
    try:
        if pdf_bytes:
            pdf_doc = fitz.open("pdf", pdf_bytes) # 从字节流加载PDF
            # 进行一些基础的优化：重新组织对象流，清除未使用的对象
            # PyMuPDF的save方法默认会进行一些清理和优化
            # garbage=4会进行更彻底的清理，deflate=True会压缩流
            # linear=True 用于Web优化的线性化
            pdf_doc.save(output_pdf_path, garbage=4, deflate=True, linear=True)
            pdf_doc.close()
            logger.info(f"PDF已优化并保存到: {output_pdf_path}")
            return True
    except Exception as e:
        logger.error(f"优化PDF {output_pdf_path} 失败: {e}", exc_info=True)
    return False


# --- Refactored functions for API integration ---

def optimize_pdf_bytes(pdf_bytes):
    """
    Optimizes PDF bytes using PyMuPDF.
    Takes raw PDF bytes as input and returns optimized PDF bytes.
    If optimization fails, logs the error and returns the original pdf_bytes.
    """
    if not pdf_bytes:
        logger.warning("optimize_pdf_bytes received None, returning None.")
        return None
    try:
        pdf_doc = fitz.open("pdf", pdf_bytes)  # Load PDF from bytes
        output_stream = io.BytesIO()
        # PyMuPDF's save options: garbage, deflate, linear. clean is not a direct save option.
        pdf_doc.save(output_stream, garbage=4, deflate=True, linear=True)
        pdf_doc.close()
        optimized_bytes = output_stream.getvalue()
        logger.info("PDF bytes optimized successfully.")
        return optimized_bytes
    except Exception as e:
        logger.error(f"Optimizing PDF bytes failed: {e}", exc_info=True)
        return pdf_bytes # Return original bytes on failure

def generate_pdf_for_api(tiff_file_path: str, config_data: dict) -> bytes | None:
    """
    Main interface for API to generate PDF from a single TIFF file.
    Takes tiff_file_path and full config_data.
    Returns PDF bytes or None if conversion fails.
    """
    logger.info(f"API call: Generating PDF for {tiff_file_path}")
    if not os.path.exists(tiff_file_path):
        logger.error(f"TIFF file not found: {tiff_file_path}")
        return None

    conversion_settings = config_data.get("conversion_settings", {})

    # Ensure Image.MAX_IMAGE_PIXELS is set (important if this function is called standalone)
    # This might be better placed at the application entry point (e.g., app.py)
    # but including it here for safety if this function is used more directly.
    if Image.MAX_IMAGE_PIXELS is None: # Check if it's not already set by main()
        logger.warning("Image.MAX_IMAGE_PIXELS was None, setting to default None (no limit) for API call.")
        Image.MAX_IMAGE_PIXELS = None


    pdf_bytes = convert_single_tiff_to_pdf_bytes(tiff_file_path, conversion_settings)

    if pdf_bytes is None:
        logger.error(f"Failed to convert TIFF to PDF bytes for {tiff_file_path}")
        return None

    if conversion_settings.get("optimize_pdf_structure", True):
        logger.info(f"Optimization enabled for {tiff_file_path}. Optimizing PDF bytes...")
        optimized_bytes = optimize_pdf_bytes(pdf_bytes)
        if optimized_bytes is pdf_bytes: # Check if optimization failed and returned original
            logger.warning(f"Optimization may have failed for {tiff_file_path}; using unoptimized or partially optimized bytes.")
        else:
            logger.info(f"PDF bytes successfully optimized for {tiff_file_path}.")
        return optimized_bytes
    else:
        logger.info(f"Optimization disabled for {tiff_file_path}. Returning original PDF bytes.")
        return pdf_bytes

# --- Modified process_file to use generate_pdf_for_api ---
def process_file(tiff_path, config):
    """
    处理单个TIFF文件的完整流程：调用 generate_pdf_for_api 并保存结果。
    This function is used by the command-line interface (main).
    """
    output_dir = config["output_directory"]
    base_name = os.path.splitext(os.path.basename(tiff_path))[0]
    output_pdf_path = os.path.join(output_dir, f"{base_name}.pdf")

    start_time = time.time()
    logger.info(f"CLI: Starting processing for: {tiff_path}")

    # Call the new API-centric function
    # The config object itself is passed as config_data
    final_pdf_bytes = generate_pdf_for_api(tiff_path, config)

    if final_pdf_bytes:
        try:
            with open(output_pdf_path, "wb") as f_out:
                f_out.write(final_pdf_bytes)
            duration = time.time() - start_time
            logger.info(f"CLI: Successfully generated and saved PDF to {output_pdf_path} (耗时: {duration:.2f} 秒)")
            return output_pdf_path, True
        except Exception as e:
            logger.error(f"CLI: Saving PDF to {output_pdf_path} failed: {e}", exc_info=True)
            return tiff_path, False # Return original tiff_path on save error
    else:
        logger.error(f"CLI: Failed to generate PDF bytes for {tiff_path}. No file saved.")
        return tiff_path, False # Return original tiff_path on generation error


# --- 3. 主程序和并行处理 ---
def main():
    try:
        config = load_config()
    except Exception:
        return # 如果配置加载失败，则退出

    setup_logging(config.get("logging_settings", {}))

    input_dir = config["input_directory"]
    output_dir = config["output_directory"]
    log_dir = config.get("log_directory", "logs") # 从配置读取日志目录

    # 创建输出和日志目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True) # 创建日志目录

    # 如果日志配置中指定了文件处理器，可以在这里设置
    # 例如，添加一个FileHandler到根记录器
    file_handler_path = os.path.join(log_dir, "conversion_log.txt")
    file_handler = logging.FileHandler(file_handler_path)
    log_format = config.get("logging_settings", {}).get("format", "%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler) # 添加到根记录器，捕获所有日志

    logger.info("--- TIFF转PDF脚本开始 ---")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"日志目录: {log_dir}")


    if not os.path.isdir(input_dir):
        logger.error(f"输入目录 {input_dir} 不存在或不是一个目录。")
        return

    tiff_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                  if f.lower().endswith(('.tif', '.tiff')) and os.path.isfile(os.path.join(input_dir, f))]

    if not tiff_files:
        logger.info("在输入目录中没有找到TIFF文件。")
        return

    logger.info(f"找到 {len(tiff_files)} 个TIFF文件待处理。")

    conversion_settings = config.get("conversion_settings", {})
    parallel_processing = conversion_settings.get("parallel_processing", True)
    num_workers = conversion_settings.get("num_workers")
    if num_workers is None or not isinstance(num_workers, int) or num_workers <= 0:
        num_workers = cpu_count() # 默认使用所有CPU核心

    total_files = len(tiff_files)
    successful_conversions = 0
    failed_conversions = 0

    start_total_time = time.time()

    if parallel_processing and len(tiff_files) > 1 : # 只有一个文件时，没必要并行
        logger.info(f"使用并行处理，工作进程数: {num_workers}")
        # 创建一个偏函数，固定 config 参数
        process_file_partial = partial(process_file, config=config)
        with Pool(processes=num_workers) as pool:
            results = list(pool.imap_unordered(process_file_partial, tiff_files))
            for result_path, success in results:
                if success:
                    successful_conversions += 1
                else:
                    failed_conversions +=1
                    logger.error(f"处理失败的文件（或原始路径）: {result_path}")

    else:
        logger.info("使用串行处理。")
        for tiff_file in tiff_files:
            result_path, success = process_file(tiff_file, config)
            if success:
                successful_conversions += 1
            else:
                failed_conversions += 1
                logger.error(f"处理失败的文件（或原始路径）: {result_path}")


    total_duration = time.time() - start_total_time
    logger.info("--- TIFF转PDF脚本结束 ---")
    logger.info(f"总文件数: {total_files}")
    logger.info(f"成功转换: {successful_conversions}")
    logger.info(f"失败转换: {failed_conversions}")
    logger.info(f"总耗时: {total_duration:.2f} 秒")

    # 移除文件处理器，以确保日志文件被正确关闭
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()

if __name__ == '__main__':
    # 确保Pillow能处理大型图像，防止解压炸弹警告
    Image.MAX_IMAGE_PIXELS = None # 设置为None表示不限制，或者可以设置为一个非常大的数字
    main()
