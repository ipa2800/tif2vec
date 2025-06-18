import os
import json
import logging
import time
import io
import argparse
import requests # Added for API hooks
from multiprocessing import Pool, cpu_count
from functools import partial
from PIL import Image, TiffTags
import img2pdf
import fitz # PyMuPDF

logger = logging.getLogger(__name__)

# --- 1. 配置和日志设置 ---
def load_config(config_path='tif2pdf_config_v2.json'):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logger.error(f"配置文件 {config_path} 未找到。", exc_info=True)
        raise
    except json.JSONDecodeError:
        logger.error(f"配置文件 {config_path} 格式错误。", exc_info=True)
        raise

def setup_logging(logging_settings):
    log_level = logging_settings.get("level", "INFO").upper()
    log_format = logging_settings.get("format", "%(asctime)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=log_level, format=log_format)

# --- 2. 核心转换逻辑 ---
def get_tiff_resolution_and_dimensions(tiff_path):
    try:
        with Image.open(tiff_path) as img:
            x_resolution_tag = img.tag.get(282)
            y_resolution_tag = img.tag.get(283)
            resolution_unit_tag = img.tag.get(296)
            dpi_x, dpi_y = None, None

            if x_resolution_tag and y_resolution_tag and resolution_unit_tag:
                if not (isinstance(x_resolution_tag, tuple) and x_resolution_tag and
                        isinstance(x_resolution_tag[0], tuple) and len(x_resolution_tag[0]) == 2 and
                        isinstance(y_resolution_tag, tuple) and y_resolution_tag and
                        isinstance(y_resolution_tag[0], tuple) and len(y_resolution_tag[0]) == 2 and
                        resolution_unit_tag[0] in [1, 2, 3]):
                    logger.warning(f"文件 {tiff_path} 的TiffTags分辨率或单位格式不符合预期。")
                else:
                    x_res_num, x_res_den = x_resolution_tag[0]
                    y_res_num, y_res_den = y_resolution_tag[0]
                    if x_res_den == 0 or y_res_den == 0:
                        logger.warning(f"文件 {tiff_path} 的TiffTags分辨率分母为零。")
                    else:
                        x_res = x_res_num / x_res_den
                        y_res = y_res_num / y_res_den
                        unit = resolution_unit_tag[0]
                        if unit == 2: dpi_x, dpi_y = x_res, y_res
                        elif unit == 3: dpi_x, dpi_y = x_res * 2.54, y_res * 2.54

            if dpi_x is None or dpi_y is None:
                pil_dpi = img.info.get('dpi')
                if pil_dpi and isinstance(pil_dpi, tuple) and len(pil_dpi) == 2 and \
                   isinstance(pil_dpi[0], (int, float)) and isinstance(pil_dpi[1], (int, float)) and \
                   pil_dpi[0] > 0 and pil_dpi[1] > 0:
                    dpi_x, dpi_y = pil_dpi
                else:
                    logger.warning(f"无法从 {tiff_path} 的TiffTags或info确定DPI，使用默认 (72, 72)。")
                    dpi_x, dpi_y = 72, 72
            return (dpi_x, dpi_y), img.size
    except FileNotFoundError:
        logger.error(f"获取分辨率失败：文件 {tiff_path} 未找到。", exc_info=True)
        return (72, 72), (0, 0)
    except Image.UnidentifiedImageError:
        logger.error(f"获取分辨率失败：Pillow无法识别 {tiff_path} 或文件损坏。", exc_info=True)
        return (72, 72), (0, 0)
    except (TypeError, AttributeError, IndexError, ZeroDivisionError) as e:
        logger.error(f"读取 {tiff_path} TiffTag元数据时内部错误: {e}。", exc_info=True)
        return (72, 72), (0, 0)
    except Exception as e:
        logger.error(f"读取 {tiff_path} 元数据时未知Pillow错误: {e}", exc_info=True)
        return (72, 72), (0, 0)

def should_downsample(original_dpi, target_dpi, dimensions, max_dimension_pixels=8000):
    if original_dpi[0] > target_dpi or original_dpi[1] > target_dpi:
        logger.info(f"原始DPI {original_dpi} 高于目标DPI {target_dpi}，需下采样。")
        return True
    if dimensions[0] > max_dimension_pixels or dimensions[1] > max_dimension_pixels:
        logger.info(f"图像尺寸 {dimensions} 超限 {max_dimension_pixels}，需下采样。")
        return True
    return False

def downsample_image(image, target_dpi):
    try:
        original_dpi = image.info.get('dpi', (target_dpi, target_dpi))
        if original_dpi[0] == 0 or original_dpi[1] == 0:
            logger.warning(f"图像原始DPI为零，使用目标DPI {target_dpi} 计算。")
            original_dpi = (target_dpi, target_dpi)

        scale_factor_x = target_dpi / original_dpi[0]
        scale_factor_y = target_dpi / original_dpi[1]
        scale_factor = min(scale_factor_x, scale_factor_y)

        if scale_factor < 1.0:
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            logger.info(f"下采样图像从 {image.size} (DPI: {original_dpi}) 到 {(new_width, new_height)} (目标DPI: {target_dpi})")
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            resized_image.info['dpi'] = (target_dpi, target_dpi)
            return resized_image
        else:
            if 'dpi' not in image.info or not (isinstance(image.info['dpi'], tuple) and len(image.info['dpi']) == 2 and image.info['dpi'][0] > 0 and image.info['dpi'][1] > 0) :
                 image.info['dpi'] = original_dpi if original_dpi[0] > 0 and original_dpi[1] > 0 else (target_dpi,target_dpi)
            return image
    except ZeroDivisionError as e:
        logger.error(f"下采样时除零错误（原始DPI可能为0）：{e}", exc_info=True)
        image.info['dpi'] = (target_dpi, target_dpi)
        return image
    except Exception as e:
        logger.error(f"下采样时Pillow操作错误: {e}", exc_info=True)
        return image

def convert_single_tiff_to_pdf_bytes(tiff_path, conversion_settings):
    img_bytes_list = []
    processed_frames = 0
    try:
        target_dpi = conversion_settings.get("target_dpi", 150)
        jpeg_quality = conversion_settings.get("jpeg_quality", 85)
        downsample_enabled = conversion_settings.get("downsample_images", True)

        with Image.open(tiff_path) as img:
            for i in range(img.n_frames):
                try:
                    img.seek(i)
                    current_image = img.copy()
                    if current_image.mode == 'P':
                        current_image = current_image.convert('RGB')
                    elif current_image.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', current_image.size, (255, 255, 255))
                        alpha_channel = current_image.split()[-1]
                        background.paste(current_image, mask=alpha_channel)
                        current_image = background

                    original_dpi_frame, dimensions_frame = get_tiff_resolution_and_dimensions(tiff_path)
                    if original_dpi_frame[0] == 0 or original_dpi_frame[1] == 0:
                        original_dpi_frame = (target_dpi, target_dpi)

                    if downsample_enabled and should_downsample(original_dpi_frame, target_dpi, dimensions_frame):
                        current_image = downsample_image(current_image, target_dpi)
                    else:
                        if 'dpi' not in current_image.info or not (isinstance(current_image.info['dpi'], tuple) and len(current_image.info['dpi']) == 2 and current_image.info['dpi'][0] > 0 and current_image.info['dpi'][1] > 0):
                            current_image.info['dpi'] = original_dpi_frame

                    frame_bytes = io.BytesIO()
                    save_format = 'JPEG'
                    if current_image.mode == '1': save_format = 'PNG'

                    current_image.save(frame_bytes, format=save_format, quality=jpeg_quality if save_format == 'JPEG' else None, dpi=current_image.info.get('dpi'), subsampling=0 if save_format == 'JPEG' else None)
                    img_bytes_list.append(frame_bytes.getvalue())
                    processed_frames += 1
                except EOFError:
                    logger.error(f"TIFF文件 {tiff_path} 超出范围的帧 {i+1}。", exc_info=True)
                    break
                except Image.DecompressionBombError as e_bomb_frame:
                    logger.error(f"处理TIFF {tiff_path} 第 {i+1}/{img.n_frames} 帧时解压炸弹错误: {e_bomb_frame}。跳过此帧。", exc_info=True)
                    continue
                except (OSError, IOError, ValueError, TypeError) as e_frame:
                    logger.error(f"处理TIFF {tiff_path} 第 {i+1}/{img.n_frames} 帧时Pillow错误: {e_frame}。跳过此帧。", exc_info=True)
                    continue
                except Exception as e_frame_unknown:
                    logger.error(f"处理TIFF {tiff_path} 第 {i+1}/{img.n_frames} 帧时未知错误: {e_frame_unknown}。跳过此帧。", exc_info=True)
                    continue
        if not img_bytes_list or processed_frames == 0:
            logger.warning(f"未能从TIFF {tiff_path} 成功提取任何可处理帧。")
            return None
        pdf_bytes = img2pdf.convert(img_bytes_list)
        logger.info(f"成功将 {tiff_path} ({processed_frames} 帧) 转为PDF字节流。")
        return pdf_bytes
    except FileNotFoundError:
        logger.error(f"转换失败：文件 {tiff_path} 未找到。", exc_info=True)
        return None
    except Image.UnidentifiedImageError:
        logger.error(f"转换失败：Pillow无法识别 {tiff_path} 或文件损坏。", exc_info=True)
        return None
    except Image.DecompressionBombError as e_bomb:
        logger.error(f"处理TIFF {tiff_path} 时解压炸弹错误: {e_bomb}。", exc_info=True)
        return None
    except img2pdf.PdfTooBigError:
        logger.error(f"转换TIFF {tiff_path} 到PDF时，PDF过大超出img2pdf限制。", exc_info=True)
        return None
    except (OSError, IOError) as e_io:
        logger.error(f"转换TIFF {tiff_path} 时文件I/O或Pillow错误: {e_io}", exc_info=True)
        return None
    except Exception as e_general:
        logger.error(f"转换TIFF {tiff_path} 到PDF时未知错误: {e_general}", exc_info=True)
        return None

# --- 3. 外部API钩子和核心API/CLI辅助函数 ---

def _replace_placeholders(payload_template: any, context_data: dict) -> any:
    if isinstance(payload_template, dict):
        return {k: _replace_placeholders(v, context_data) for k, v in payload_template.items()}
    elif isinstance(payload_template, list):
        return [_replace_placeholders(item, context_data) for item in payload_template]
    elif isinstance(payload_template, str):
        processed_string = payload_template
        for key, value in context_data.items():
            placeholder = "{" + str(key) + "}"
            processed_string = processed_string.replace(placeholder, str(value))
        return processed_string
    else:
        return payload_template

def execute_api_hook(hook_name: str, config_data: dict, context_data: dict) -> bool:
    hooks_config = config_data.get("external_api_hooks", {})
    hook_config = hooks_config.get(hook_name)

    if not hook_config or not hook_config.get("enabled", False):
        logger.debug(f"API钩子 '{hook_name}' 未配置或已禁用。")
        return True

    logger.info(f"执行API钩子: '{hook_name}'，URL: {hook_config.get('url')}")
    url = hook_config.get("url")
    method = hook_config.get("method", "POST").upper()
    headers = hook_config.get("headers", {})
    timeout = hook_config.get("timeout_seconds", 10)
    payload_type = hook_config.get("payload_to_send", "custom_json")
    custom_payload_template = hook_config.get("custom_json_payload", {})
    blocking = hook_config.get("blocking", True)
    success_codes = hook_config.get("success_http_codes", [200])
    retries = hook_config.get("retry_attempts", 0)
    retry_delay = hook_config.get("retry_delay_seconds", 5)

    full_context = context_data.copy()
    if "tiff_file_path" in full_context and full_context["tiff_file_path"] and os.path.exists(full_context["tiff_file_path"]):
        full_context.setdefault("file_name", os.path.basename(full_context["tiff_file_path"]))
        try:
            full_context.setdefault("file_size_bytes", os.path.getsize(full_context["tiff_file_path"]))
        except OSError as e_size:
            logger.warning(f"钩子 '{hook_name}': 无法获取文件 {full_context['tiff_file_path']} 大小: {e_size}。")
            full_context.setdefault("file_size_bytes", -1)
    else:
        full_context.setdefault("file_name", context_data.get("original_file_name", "unknown_file"))
        full_context.setdefault("file_size_bytes", -1)
    if "pdf_bytes" in full_context and isinstance(full_context["pdf_bytes"], bytes):
        full_context.setdefault("pdf_size_bytes", len(full_context["pdf_bytes"]))
    else:
        full_context.setdefault("pdf_size_bytes", 0)
    full_context.setdefault("file_name_original_tiff", context_data.get("original_file_name", full_context.get("file_name")))

    json_payload = None
    if method in ["POST", "PUT", "PATCH"]:
        if payload_type == "file_path":
            json_payload = {"file_path": full_context.get("tiff_file_path"), "file_name": full_context.get("file_name")}
        elif payload_type == "file_name":
            json_payload = {"file_name": full_context.get("original_file_name")}
        elif payload_type == "pdf_bytes_metadata":
            json_payload = {
                "original_tiff_path": full_context.get("tiff_file_path"),
                "original_tiff_name": full_context.get("file_name_original_tiff"),
                "pdf_size_bytes": full_context.get("pdf_size_bytes", 0),
            }
        elif payload_type == "custom_json":
            json_payload = _replace_placeholders(custom_payload_template, full_context)
        else:
            logger.warning(f"钩子 '{hook_name}': 未知 payload_to_send 类型 '{payload_type}'。")
            if method in ["POST", "PUT", "PATCH"]: json_payload = {}

    if not url:
        logger.error(f"钩子 '{hook_name}' 配置不完整：缺少URL。")
        return not blocking

    last_exception = None
    for attempt in range(retries + 1):
        try:
            logger.debug(f"钩子 '{hook_name}': 尝试 {attempt + 1}/{retries + 1} {method} {url}。负载: {json_payload}")
            current_headers = headers.copy()
            if json_payload is not None:
                if "application/json" in current_headers.get("Content-Type", "").lower():
                    response = requests.request(method, url, headers=current_headers, json=json_payload, timeout=timeout)
                else:
                    response = requests.request(method, url, headers=current_headers, data=json_payload, timeout=timeout)
            else:
                 response = requests.request(method, url, headers=current_headers, timeout=timeout)
            logger.info(f"钩子 '{hook_name}' API响应: {response.status_code} (尝试 {attempt + 1})")
            if response.status_code in success_codes:
                logger.info(f"钩子 '{hook_name}' 成功。")
                logger.debug(f"钩子 '{hook_name}' 响应内容: {response.text[:500]}")
                return True
            else:
                logger.error(f"钩子 '{hook_name}' API失败。状态: {response.status_code}, 响应: {response.text[:500]}")
                last_exception = Exception(f"API返回失败状态: {response.status_code}")
                if attempt < retries: time.sleep(retry_delay)
                else: break
        except requests.exceptions.Timeout as e_timeout:
            logger.error(f"钩子 '{hook_name}' API超时 ({timeout}s) {url}: {e_timeout}", exc_info=True)
            last_exception = e_timeout
            if attempt < retries: time.sleep(retry_delay)
            else: break
        except requests.exceptions.RequestException as e_req:
            logger.error(f"钩子 '{hook_name}' API请求错误 ({url}): {e_req}", exc_info=True)
            last_exception = e_req
            if attempt < retries: time.sleep(retry_delay)
            else: break
        except Exception as e_unknown:
            logger.error(f"钩子 '{hook_name}' 执行时 ({url}) 未知错误: {e_unknown}", exc_info=True)
            last_exception = e_unknown
            break
    if last_exception is not None:
        if blocking:
            logger.critical(f"阻塞API钩子 '{hook_name}' ({retries + 1}次尝试)失败: {last_exception}")
            return False
        else:
            logger.warning(f"非阻塞API钩子 '{hook_name}' ({retries + 1}次尝试)失败: {last_exception}。继续...")
            return True
    return True

def optimize_pdf_bytes(pdf_bytes):
    if not pdf_bytes:
        logger.warning("optimize_pdf_bytes接收到None，返回原始（空）字节流。")
        return pdf_bytes
    doc = None
    try:
        doc = fitz.open("pdf", pdf_bytes)
        output_stream = io.BytesIO()
        doc.save(output_stream, garbage=4, deflate=True, linear=True)
        optimized_bytes = output_stream.getvalue()
        logger.info("PDF字节流已用PyMuPDF优化。")
        return optimized_bytes
    except fitz.fitz.FZ_ERROR_GENERIC as e_fitz_generic:
        logger.error(f"PyMuPDF优化PDF字节流FZ_ERROR_GENERIC: {e_fitz_generic}。返回原始字节。", exc_info=True)
        return pdf_bytes
    except RuntimeError as e_runtime:
        logger.error(f"PyMuPDF优化PDF字节流RuntimeError: {e_runtime}。返回原始字节。", exc_info=True)
        return pdf_bytes
    except Exception as e_unknown:
        logger.error(f"PyMuPDF优化PDF字节流未知错误: {e_unknown}。返回原始字节。", exc_info=True)
        return pdf_bytes
    finally:
        if doc:
            try: doc.close()
            except Exception as e_close: logger.error(f"关闭PyMuPDF文档时错误: {e_close}", exc_info=True)

def generate_pdf_for_api(tiff_file_path: str, config_data: dict) -> bytes | None:
    logger.info(f"开始为TIFF文件 {tiff_file_path} 生成PDF。")
    hook_context_before = {
        "tiff_file_path": tiff_file_path,
        "original_file_name": os.path.basename(tiff_file_path)
    }
    if not execute_api_hook("on_before_conversion_start", config_data, hook_context_before):
        logger.error(f"阻塞API钩子 'on_before_conversion_start' 失败: {tiff_file_path}。中止。")
        return None
    if not os.path.exists(tiff_file_path):
        logger.error(f"TIFF文件 {tiff_file_path} 在 'on_before_conversion_start' 钩子后未找到。")
        return None

    conversion_settings = config_data.get("conversion_settings", {})
    if Image.MAX_IMAGE_PIXELS is None:
        logger.warning("Image.MAX_IMAGE_PIXELS未设置，临时设为None。")
        Image.MAX_IMAGE_PIXELS = None

    pdf_bytes = convert_single_tiff_to_pdf_bytes(tiff_file_path, conversion_settings)
    if pdf_bytes is None:
        logger.error(f"未能将TIFF {tiff_file_path} 转为PDF字节流。")
        return None

    final_pdf_bytes = pdf_bytes
    if conversion_settings.get("optimize_pdf_structure", True):
        logger.info(f"为 {tiff_file_path} 启用PDF优化。")
        optimized_bytes = optimize_pdf_bytes(pdf_bytes)
        if optimized_bytes is pdf_bytes and pdf_bytes is not None :
            logger.warning(f"PDF优化可能失败或未产生更改 {tiff_file_path}。")
        elif optimized_bytes is not None:
            logger.info(f"成功优化 {tiff_file_path} PDF。")
        if optimized_bytes is not None:
             final_pdf_bytes = optimized_bytes
    else:
        logger.info(f"为 {tiff_file_path} 禁用PDF优化。")

    hook_context_after = {
        "tiff_file_path": tiff_file_path,
        "original_file_name": os.path.basename(tiff_file_path),
        "pdf_bytes": final_pdf_bytes,
    }
    if not execute_api_hook("on_after_pdf_generation", config_data, hook_context_after):
        hook_after_config = config_data.get("external_api_hooks", {}).get("on_after_pdf_generation", {})
        if hook_after_config.get("blocking", False):
            logger.error(f"阻塞API钩子 'on_after_pdf_generation' 失败: {tiff_file_path}。")
            return None
        else:
            logger.warning(f"非阻塞API钩子 'on_after_pdf_generation' 失败: {tiff_file_path}。继续。")
    return final_pdf_bytes

# --- 4. 命令行界面 (CLI) 和主程序 ---
def process_file(tiff_path, config):
    output_dir = config["output_directory"]
    base_name = os.path.splitext(os.path.basename(tiff_path))[0]
    output_pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    start_time = time.time()
    logger.info(f"CLI：开始处理: {tiff_path}")
    final_pdf_bytes = generate_pdf_for_api(tiff_path, config)
    if final_pdf_bytes:
        try:
            with open(output_pdf_path, "wb") as f_out:
                f_out.write(final_pdf_bytes)
            duration = time.time() - start_time
            logger.info(f"CLI：成功保存PDF到 {output_pdf_path} (耗时: {duration:.2f}s)")
            return output_pdf_path, True
        except IOError as e_io:
            logger.error(f"CLI：保存PDF到 {output_pdf_path} IOError: {e_io}", exc_info=True)
            return tiff_path, False
        except Exception as e_save:
            logger.error(f"CLI：保存PDF到 {output_pdf_path} 未知错误: {e_save}", exc_info=True)
            return tiff_path, False
    else:
        logger.error(f"CLI：为 {tiff_path} 生成PDF字节流失败。")
        return tiff_path, False

# --- 5. 主程序和并行处理 ---
def main():
    parser = argparse.ArgumentParser(description="TIFF转PDF转换器。")
    parser.add_argument("input_path", help="输入TIFF文件或目录路径。")
    parser.add_argument("--output_dir", "-o", default=None, help="输出PDF目录。覆盖配置。")
    parser.add_argument("--config_file", "-c", default="tif2pdf_config_v2.json", help="配置文件路径。")
    parser.add_argument("--no-parallel", action="store_false", dest="parallel_processing", help="禁用并行处理。")
    parser.set_defaults(parallel_processing=None)
    parser.add_argument("--num-workers", "-w", type=int, default=None, help="并行转换的工作进程数。")
    parser.add_argument("--no-downsample", action="store_false", dest="downsample_images", help="禁用图像下采样。")
    parser.set_defaults(downsample_images=None)
    parser.add_argument("--dpi", type=int, default=None, help="图像下采样目标DPI。")
    parser.add_argument("--quality", type=int, default=None, help="JPEG质量 (1-100)。")
    parser.add_argument("--no-optimize", action="store_false", dest="optimize_pdf_structure", help="禁用PDF优化。")
    parser.set_defaults(optimize_pdf_structure=None)
    args = parser.parse_args()

    try:
        config = load_config(args.config_file)
        logger.info(f"从 {args.config_file} 加载配置。")
    except Exception as e:
        print(f"错误: 无法加载配置文件 '{args.config_file}': {e}。退出。")
        logging.error(f"无法加载配置文件 '{args.config_file}': {e}", exc_info=True)
        return 1

    if args.output_dir: config["output_directory"] = args.output_dir
    if "conversion_settings" not in config or not isinstance(config["conversion_settings"], dict):
        config["conversion_settings"] = {}

    if args.parallel_processing is False: config["conversion_settings"]["parallel_processing"] = False
    if args.num_workers is not None: config["conversion_settings"]["num_workers"] = args.num_workers
    if args.downsample_images is False: config["conversion_settings"]["downsample_images"] = False
    if args.dpi is not None: config["conversion_settings"]["target_dpi"] = args.dpi
    if args.quality is not None:
        if 1 <= args.quality <= 100: config["conversion_settings"]["jpeg_quality"] = args.quality
        else: logger.warning(f"JPEG质量值 {args.quality} 无效。忽略。")
    if args.optimize_pdf_structure is False: config["conversion_settings"]["optimize_pdf_structure"] = False

    try:
        setup_logging(config.get("logging_settings", {}))
    except Exception as e_log_setup:
        print(f"错误: 配置日志系统失败: {e_log_setup}")
        logging.basicConfig(level=logging.WARNING)
        logger.error(f"配置日志系统失败: {e_log_setup}", exc_info=True)

    try:
        output_dir = config["output_directory"]
        log_dir = config.get("log_directory", "logs")
    except KeyError as e_key_config:
        logger.error(f"最终配置缺少目录项: {e_key_config}。退出。", exc_info=True)
        return 1
    try:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e_mkdir:
        logger.error(f"创建目录 '{output_dir}' 或 '{log_dir}' 失败: {e_mkdir}。退出。", exc_info=True)
        return 1

    file_handler = None
    try:
        file_handler_path = os.path.join(log_dir, "conversion_cli_log.txt")
        file_handler = logging.FileHandler(file_handler_path, encoding='utf-8')
        log_format_cfg = config.get("logging_settings", {}).get("format", "%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(logging.Formatter(log_format_cfg))
        logging.getLogger().addHandler(file_handler)
    except Exception as e_fh:
        logger.error(f"无法创建日志文件处理器 {file_handler_path}: {e_fh}", exc_info=True)

    logger.info(f"--- TIFF到PDF CLI转换开始 ---")
    logger.info(f"输入路径: {args.input_path}")
    logger.info(f"最终配置: {json.dumps(config, indent=4, ensure_ascii=False)}")

    tiff_files = []
    if os.path.isfile(args.input_path):
        if args.input_path.lower().endswith(('.tif', '.tiff')): tiff_files.append(args.input_path)
        else: logger.error(f"输入文件 '{args.input_path}' 非TIFF。退出。")
    elif os.path.isdir(args.input_path):
        logger.info(f"扫描目录 '{args.input_path}'...")
        try:
            tiff_files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path)
                          if f.lower().endswith(('.tif', '.tiff')) and os.path.isfile(os.path.join(args.input_path, f))]
        except Exception as e_scan: logger.error(f"扫描目录 '{args.input_path}' 错误: {e_scan}。退出。", exc_info=True)
    else: logger.error(f"输入路径 '{args.input_path}' 无效。退出。")

    if not tiff_files:
        logger.info(f"路径 '{args.input_path}' 无TIFF文件。结束。")
        if file_handler: logging.getLogger().removeHandler(file_handler); file_handler.close()
        return 0

    logger.info(f"找到 {len(tiff_files)} 个TIFF文件处理。")

    final_conv_settings = config.get("conversion_settings", {})
    parallel = final_conv_settings.get("parallel_processing", True)
    num_w = final_conv_settings.get("num_workers")
    if num_w is None or not isinstance(num_w, int) or num_w <= 0: num_w = cpu_count()
    final_conv_settings["num_workers"] = num_w

    total_files = len(tiff_files)
    success_count = 0
    fail_count = 0
    total_start_time = time.time()

    if parallel and total_files > 1:
        logger.info(f"并行处理 {total_files} 文件，{num_w} 工作进程。")
        p_process = partial(process_file, config=config)
        try:
            with Pool(processes=num_w) as pool:
                results = list(pool.imap_unordered(p_process, tiff_files))
            for _, success_flag in results:
                if success_flag: success_count += 1
                else: fail_count += 1
        except Exception as e_pool:
            logger.error(f"并行处理严重错误: {e_pool}", exc_info=True)
            fail_count = total_files - success_count
    else:
        logger.info("串行处理。")
        for tiff_f_path in tiff_files:
            _, success_flag = process_file(tiff_f_path, config)
            if success_flag: success_count += 1
            else: fail_count += 1

    logger.info(f"--- TIFF到PDF CLI转换结束 ---")
    logger.info(f"总文件: {total_files}, 成功: {success_count}, 失败: {fail_count}")
    logger.info(f"总耗时: {time.time() - total_start_time:.2f}s")

    if file_handler:
        try:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
        except Exception as e_fh_close: logger.error(f"关闭日志处理器错误: {e_fh_close}", exc_info=True)
    return 0 if fail_count == 0 else 1

if __name__ == '__main__':
    try:
        Image.MAX_IMAGE_PIXELS = None
    except NameError:
        print("[警告] Pillow库Image模块未找到。MAX_IMAGE_PIXELS未设置。")
    except Exception as e_pil_max:
        print(f"[警告] 设置Pillow Image.MAX_IMAGE_PIXELS错误: {e_pil_max}")
    import sys

    # --- Temporary Test Block for API Hooks ---
    def _run_hook_tests():
        print("\n--- RUNNING API HOOK TESTS ---")
        # Ensure the test_tiffs directory and dummy files exist as created in a previous step
        # For this test, we assume they are in the current working directory or a known path.
        # If using a relative path like "test_tiffs/dummy1.tif", ensure CWD is the project root.
        module_dir = os.path.dirname(__file__) if os.path.dirname(__file__) else "."
        dummy_tiff_path = os.path.join(module_dir, "test_tiffs/dummy1.tif")

        if not os.path.exists(dummy_tiff_path):
            print(f"ERROR: Dummy TIFF file {dummy_tiff_path} not found for testing. Skipping hook tests.")
            # Attempt to create it if it's missing, for a slightly more robust test run
            try:
                os.makedirs(os.path.dirname(dummy_tiff_path), exist_ok=True)
                with open(dummy_tiff_path, 'w') as f: f.write("dummy tiff content") # Minimal content
                print(f"INFO: Created dummy file {dummy_tiff_path} for test.")
            except Exception as e_create:
                print(f"ERROR: Could not create dummy tiff for testing: {e_create}")
                return

        base_config = {
            "conversion_settings": {
                "optimize_pdf_structure": False
            },
            "logging_settings": {"level": "DEBUG"},
            "external_api_hooks": {}
        }
        # It's crucial that setup_logging is called for these tests to see log output
        # if the main script's logging isn't already configured when this block runs.
        # However, main() usually sets up logging. If running this standalone, ensure it's called.
        # For safety here, re-initialize basicConfig if no handlers, or use existing.
        if not logging.getLogger().hasHandlers():
             setup_logging(base_config["logging_settings"])
        logger.info("Hook test logging is active.")


        # Test Case 1: on_before_conversion_start, blocking=true, API fails
        print("\n[Test Case 1: before_conversion, blocking=true, API fails]")
        config1 = json.loads(json.dumps(base_config))
        config1["external_api_hooks"]["on_before_conversion_start"] = {
            "enabled": True, "url": "http://localhost:12345/mock_fail", "method": "POST",
            "blocking": True, "retry_attempts": 1, "retry_delay_seconds": 0, "timeout_seconds": 1, # Fast retry
            "payload_to_send": "file_path", "success_http_codes": [200]
        }
        result1 = generate_pdf_for_api(dummy_tiff_path, config1)
        print(f"Test Case 1 Result: {'Conversion aborted as expected' if result1 is None else 'ERROR: Conversion NOT aborted'}")

        # Test Case 2: on_before_conversion_start, blocking=false, API fails
        print("\n[Test Case 2: before_conversion, blocking=false, API fails]")
        config2 = json.loads(json.dumps(base_config))
        config2["external_api_hooks"]["on_before_conversion_start"] = {
            "enabled": True, "url": "http://localhost:12345/mock_fail", "method": "POST",
            "blocking": False, "retry_attempts": 1, "retry_delay_seconds": 0, "timeout_seconds": 1,
            "payload_to_send": "file_name", "success_http_codes": [200]
        }
        # generate_pdf_for_api will likely return None due to dummy file, but hook logs are key.
        _ = generate_pdf_for_api(dummy_tiff_path, config2)
        print(f"Test Case 2 Result: Hook executed (check logs for 'Non-blocking API hook ... failed ... Continuing...').")

        # Test Case 3: on_after_pdf_generation, blocking=true, API fails
        # This test is tricky because generate_pdf_for_api will return None for a dummy .tif file.
        # The on_after_pdf_generation hook is only called if pdf_bytes is not None.
        # To truly test this hook's blocking=true, we'd need a minimal valid TIFF or mock convert_single_tiff_to_pdf_bytes.
        # For now, we demonstrate that the hook isn't called if pdf_bytes generation fails.
        print("\n[Test Case 3: after_generation, blocking=true, API fails - with dummy file (hook might not run)]")
        config3 = json.loads(json.dumps(base_config))
        config3["external_api_hooks"]["on_after_pdf_generation"] = {
            "enabled": True, "url": "http://localhost:12345/mock_fail", "method": "POST",
            "blocking": True, "retry_attempts": 0, "timeout_seconds": 1,
            "payload_to_send": "pdf_bytes_metadata", "success_http_codes": [200]
        }
        result3 = generate_pdf_for_api(dummy_tiff_path, config3)
        print(f"Test Case 3 Result: {'Conversion failed (as dummy file cannot be processed) or hook aborted' if result3 is None else 'ERROR: Unexpected success with dummy file'}")
        # If result3 is None, check logs to see if it was due to the hook or earlier PDF generation failure.
        # Expected: convert_single_tiff_to_pdf_bytes returns None for dummy1.tif, so after_generation hook isn't called.

        # Test Case 4: on_after_pdf_generation, blocking=false, API fails (custom payload)
        print("\n[Test Case 4: after_generation, blocking=false, API fails, custom_json - with dummy file (hook might not run)]")
        config4 = json.loads(json.dumps(base_config))
        config4["external_api_hooks"]["on_after_pdf_generation"] = {
            "enabled": True, "url": "http://localhost:12345/mock_fail", "method": "POST",
            "blocking": False, "retry_attempts": 0, "timeout_seconds": 1,
            "payload_to_send": "custom_json",
            "custom_json_payload": {"file": "{file_name}", "size": "{file_size_bytes}", "pdf_size": "{pdf_size_bytes}", "hook_event":"on_after_pdf_generation_test"},
            "success_http_codes": [200]
        }
        _ = generate_pdf_for_api(dummy_tiff_path, config4)
        print(f"Test Case 4 Result: Hook execution attempted (check logs for non-blocking failure and custom payload if PDF was generated).")

        print("\n--- API HOOK TESTS FINISHED ---")

    # To run these tests, uncomment the line below when executing tiff_converter.py directly.
    # This is primarily for development and isolated testing of the hook mechanism.
    # _run_hook_tests()

    exit_code = main()
    sys.exit(exit_code)
