import os  # 操作系统接口模块，用于文件路径和目录操作
import uuid  # UUID模块，用于生成唯一的文件名，避免冲突
import io  # 输入输出模块，用于处理内存中的字节流 (例如，发送文件响应)
from flask import Flask, request, send_file, jsonify, abort  # Flask框架及其组件
import logging  # 日志记录模块
from werkzeug.utils import secure_filename  # Werkzeug工具，用于获取安全的文件名

# 假设 tiff_converter.py 文件与 app.py 在同一个目录下
import tiff_converter  # 导入自定义的TIFF转换逻辑模块

# --- 配置与初始化 ---
# Application Configuration and Initialization Section

# 尝试创建 'uploads' 和 'logs' 目录
try:
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('logs'):
        os.makedirs('logs')
except OSError as e_mkdir:
    # 如果在应用启动时无法创建这些基本目录，记录严重错误。
    # 应用可能无法正常工作（例如，无法保存上传的文件或写入日志）。
    logging.basicConfig(level=logging.CRITICAL) # 确保此消息能被看到
    logging.critical(f"严重错误：无法创建基础目录 'uploads' 或 'logs': {e_mkdir}", exc_info=True)
    # 在生产环境中，这可能是一个需要立即停止应用的条件。
    # 对于此脚本，我们将允许它继续，但功能会受限。

DUMMY_CONFIG_USED = False # 标记是否使用了虚拟配置
try:
    CONFIG = tiff_converter.load_config('tif2pdf_config_v2.json')
    if CONFIG is None: # load_config 应该抛出异常，但这作为后备
        raise ValueError("加载配置文件失败：load_config 返回 None。")
except FileNotFoundError as e_fnf:
    logging.basicConfig(level=logging.CRITICAL)
    logging.critical(f"严重错误：配置文件 'tif2pdf_config_v2.json' 未找到: {e_fnf}", exc_info=True)
    DUMMY_CONFIG_USED = True
except json.JSONDecodeError as e_json:
    logging.basicConfig(level=logging.CRITICAL)
    logging.critical(f"严重错误：配置文件 'tif2pdf_config_v2.json' 格式错误: {e_json}", exc_info=True)
    DUMMY_CONFIG_USED = True
except Exception as e_conf_load: # 捕获其他可能的加载错误
    logging.basicConfig(level=logging.CRITICAL)
    logging.critical(f"严重错误：加载配置文件 'tif2pdf_config_v2.json' 时发生未知错误: {e_conf_load}", exc_info=True)
    DUMMY_CONFIG_USED = True

if DUMMY_CONFIG_USED:
    CONFIG = {
        "logging_settings": {"level": "WARNING", "format": "%(asctime)s - %(levelname)s - %(message)s"},
        "conversion_settings": {},
        "input_directory": "uploads", # 更改为uploads，因为这是API实际使用的
        "output_directory": "output_pdfs_dummy", # 虚拟输出
        "log_directory": "logs"
    }
    # 日志系统可能尚未完全配置，因此这个警告可能只到控制台
    logging.warning("警告：由于加载主配置文件失败，正在使用受限的虚拟配置。API 功能将严重受限。")

# 设置日志系统
try:
    tiff_converter.setup_logging(CONFIG.get('logging_settings', {"level": "INFO"})) # 提供默认值以防万一
except Exception as e_log_setup:
    logging.basicConfig(level=logging.ERROR) # 极简日志配置
    logging.error(f"配置日志系统失败: {e_log_setup}", exc_info=True)
    # 如果日志设置失败，后续的日志记录可能不按预期工作。

logger = logging.getLogger(__name__) # 使用模块特定的记录器

logger.info("Flask应用程序开始启动...")
if DUMMY_CONFIG_USED:
    logger.critical("CRITICAL: Flask应用程序正在使用虚拟配置。许多功能可能无法正常工作。请检查配置文件 'tif2pdf_config_v2.json'。")

# 检查CONFIG中是否存在必要的键
essential_keys = ["conversion_settings", "input_directory", "output_directory", "log_directory"]
missing_keys = [key for key in essential_keys if key not in CONFIG]
if missing_keys:
    logger.critical(f"CRITICAL: 配置文件中缺少以下必要的顶级键: {', '.join(missing_keys)}。API可能无法正常运行。")
    # 根据应用的严格程度，这里可以决定是否 raise SystemExit("配置不完整，应用无法启动。")

app = Flask(__name__) # 创建Flask应用实例

# --- 全局应用设置 (可从CONFIG加载) ---
# Flask Application Global Settings

# 示例：如果需要，可以从配置文件设置最大上传大小，否则使用Flask的默认值
# app.config['MAX_CONTENT_LENGTH'] = CONFIG.get('max_upload_size_bytes', 16 * 1024 * 1024) # 默认为16MB
UPLOAD_FOLDER = 'uploads' # 定义上传文件存储的文件夹
ALLOWED_EXTENSIONS = {'tif', 'tiff'} # 定义允许上传的文件扩展名集合

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # 将上传文件夹配置到Flask应用

def allowed_file(filename):
    """
    检查上传的文件名是否具有允许的扩展名。
    Args:
        filename (str): 用户上传的原始文件名。
    Returns:
        bool: 如果文件扩展名在 ALLOWED_EXTENSIONS 中，则返回 True，否则返回 False。
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- API 端点 ---
# API Endpoints Section

@app.route('/convert_tiff', methods=['POST']) # 定义路由和允许的HTTP方法
def convert_tiff_endpoint():
    """
    TIFF转PDF的API端点。
    接收一个TIFF文件，将其转换为PDF，并以文件形式返回转换后的PDF。
    HTTP方法: POST
    请求体: multipart/form-data, 包含一个名为 'tiff_file' 的文件部分。
    成功响应: 200 OK, 返回PDF文件。
    错误响应:
        400 Bad Request: 如果请求中没有文件、没有选择文件或文件类型不允许。
        500 Internal Server Error: 如果转换过程中发生内部错误。
    """
    # 1. 检查请求中是否包含 'tiff_file' 部分
    if 'tiff_file' not in request.files:
        logger.error("API错误：请求中缺少 'tiff_file' 部分。")
        return jsonify({"error": "请求中缺少 'tiff_file' 文件部分"}), 400 # 返回JSON错误信息和状态码

    file = request.files['tiff_file'] # 获取上传的文件对象

    # 2. 检查文件名是否为空 (即用户没有选择文件)
    if file.filename == '':
        logger.error("API错误：未选择任何文件。")
        return jsonify({"error": "未选择任何文件"}), 400

    # 3. 检查文件是否存在且文件类型是否允许
    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename) # 获取安全的文件名，防止路径遍历等攻击
        temp_filename = str(uuid.uuid4()) + "_" + original_filename
        temp_tiff_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)

        try:
            # 4. 保存上传的文件到服务器的临时位置
            try:
                file.save(temp_tiff_path)
                logger.info(f"上传的文件已临时保存为: {temp_tiff_path}")
            except (IOError, OSError) as e_save:
                logger.error(f"保存上传的文件 {original_filename} (到 {temp_tiff_path}) 失败: {e_save}", exc_info=True)
                return jsonify({"error": "保存上传文件失败，请检查服务器权限或磁盘空间。"}), 500

            # 5. 检查关键配置是否存在，如果使用了虚拟配置，可能无法正确转换
            if DUMMY_CONFIG_USED or "conversion_settings" not in CONFIG:
                 logger.critical(f"由于配置加载不完整或错误，无法处理文件 {original_filename} 的转换请求。")
                 return jsonify({"error": "服务器配置错误，无法处理转换请求。请联系管理员。"}), 500

            # 调用 tiff_converter 模块中的核心转换函数
            logger.info(f"开始转换TIFF文件: {original_filename} (临时文件: {temp_tiff_path})")
            pdf_bytes = None # 初始化
            try:
                pdf_bytes = tiff_converter.generate_pdf_for_api(temp_tiff_path, CONFIG)
            except Exception as e_converter: # 捕获从tiff_converter传播的未捕获异常
                logger.error(f"TIFF转换逻辑 (generate_pdf_for_api) 针对文件 {original_filename} 抛出未捕获的异常: {e_converter}", exc_info=True)
                return jsonify({"error": f"文件 {original_filename} 转换过程中发生意外的服务器内部错误。"}), 500

            if pdf_bytes:
                logger.info(f"文件 {original_filename} 已成功转换为PDF。")
                output_filename = os.path.splitext(original_filename)[0] + ".pdf"
                return send_file(
                    io.BytesIO(pdf_bytes),
                    mimetype='application/pdf',
                    as_attachment=True,
                    download_name=output_filename
                )
            else: # generate_pdf_for_api 返回 None，表示已知转换失败
                logger.error(f"文件 {original_filename} 转换失败。tiff_converter.generate_pdf_for_api 返回 None。")
                error_message = (
                    f"无法转换TIFF文件: {original_filename}。"
                    "可能的原因包括：不支持的TIFF格式、文件损坏、或内部处理错误。"
                    "详情请查看服务器日志。"
                )
                return jsonify({"error": error_message}), 500

        except Exception as e_main: # 捕获此 try 块中任何其他未预料的异常
            logger.error(f"处理文件 {original_filename} 的转换请求时发生未知错误: {e_main}", exc_info=True)
            return jsonify({"error": "处理您的请求时发生不可预知的服务器错误。"}), 500
        finally:
            # 清理临时上传的文件
            if os.path.exists(temp_tiff_path):
                try:
                    os.remove(temp_tiff_path)
                    logger.info(f"已清理临时文件: {temp_tiff_path}")
                except OSError as e_remove: # 更具体的异常
                    logger.error(f"清理临时文件 {temp_tiff_path} 时发生OS错误: {e_remove}", exc_info=True)
    else: # 文件类型不允许
        logger.warning(f"API警告：文件 {file.filename if file else 'N/A'} 的类型不被允许。")
        return jsonify({"error": "文件类型不被允许。请上传 .tif 或 .tiff 文件。"}), 400

@app.route('/health', methods=['GET']) # 健康检查端点
def health_check():
    """
    提供API的健康状态。
    HTTP方法: GET
    成功响应: 200 OK, 返回包含状态信息的JSON对象。
    """
    # 基础的健康检查端点
    # 可以扩展以检查依赖项 (如TIFF库、磁盘空间、配置是否成功加载等)
    health_status = "healthy"
    health_message = "TIFF转PDF转换器API正在运行。"
    if DUMMY_CONFIG_USED:
        health_status = "degraded"
        health_message = "TIFF转PDF转换器API正在运行，但使用的是虚拟配置，功能可能受限。"
        logger.warning("健康检查：API处于降级状态，因为使用了虚拟配置。")

    logger.info(f"健康检查请求已接收。状态: {health_status}")
    return jsonify({"status": health_status, "message": health_message}), 200

# --- 主程序入口 ---
# Main Application Entry Point

if __name__ == '__main__':
    # 确保Pillow库可以处理大型图像
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        logger.info("Pillow Image.MAX_IMAGE_PIXELS 已成功设置为 None (无限制)。")
    except ImportError:
        logger.warning("未能导入Pillow库。如果Pillow未安装，图像处理功能将不可用。")
    except Exception as e_pil_max: # 其他可能的PIL错误
        logger.error(f"设置Pillow Image.MAX_IMAGE_PIXELS时发生错误: {e_pil_max}", exc_info=True)

    # 以下用于Flask开发服务器。
    # 对于生产环境，请使用功能更完善的WSGI服务器 (如 Gunicorn, uWSGI)。
    # host='0.0.0.0' 使其可以从网络中的其他机器访问 (例如，在Docker容器内部署时)。
    # debug=True 仅用于开发模式，切勿在生产环境中使用，因为它可能暴露安全漏洞。
    logger.info("准备启动Flask开发服务器...")
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e_app_run: # 例如端口已被占用
        logger.critical(f"Flask开发服务器启动失败: {e_app_run}", exc_info=True)
        # 在这种情况下，应用无法服务请求，所以退出可能是合理的。
        # exit(1) # 或者更复杂的重启逻辑
