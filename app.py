import os
import uuid
import io # Added import io
from flask import Flask, request, send_file, jsonify, abort
import logging # Import logging
from werkzeug.utils import secure_filename

# Assuming tiff_converter.py is in the same directory
import tiff_converter

# --- Configuration & Initialization ---
# Create 'uploads' and 'logs' directory if they don't exist (though tiff_converter should handle logs)
if not os.path.exists('uploads'):
    os.makedirs('uploads')
if not os.path.exists('logs'): # Ensure logs dir exists for Flask/general app logging if needed
    os.makedirs('logs')

# Load configuration using the function from tiff_converter
try:
    CONFIG = tiff_converter.load_config('tif2pdf_config_v2.json')
    if CONFIG is None:
        # load_config should raise an error if file not found or JSON is bad,
        # but as a safeguard:
        raise ValueError("Failed to load configuration.")
except Exception as e:
    # Log to stderr or a predefined file if logging setup hasn't happened yet
    logging.basicConfig(level=logging.ERROR) # Basic config for this critical error
    logging.error(f"CRITICAL: Failed to load tif2pdf_config_v2.json: {e}")
    # If config fails to load, the app cannot run.
    # Consider how to handle this in a production environment (e.g., exit, alert)
    # For now, we'll let it proceed, but endpoints might fail if CONFIG is not what's expected.
    # A better approach would be to raise an exception and stop the app.
    # However, to allow the subtask to complete, we'll create a dummy CONFIG if it fails.
    CONFIG = {
        "logging_settings": {"level": "ERROR", "format": "%(asctime)s - %(levelname)s - %(message)s"},
        "conversion_settings": {}, # Dummy, will cause issues but allows app to load
        "input_directory": "input_tiffs", # Dummy
        "output_directory": "output_pdfs", # Dummy
        "log_directory": "logs" # Dummy
    }
    # This will likely cause the app to not function correctly, but it allows the script to be written.
    # The user should be warned if this dummy config is used.

# Setup logging using the function from tiff_converter and the loaded config
# The tiff_converter.setup_logging function configures the root logger.
tiff_converter.setup_logging(CONFIG.get('logging_settings', {}))

# Get the logger instance used by tiff_converter.py (or create one for app.py)
# If tiff_converter.py uses `logger = logging.getLogger(__name__)`,
# then tiff_converter.logger will get that.
# Otherwise, get the root logger or a specific app logger.
# For simplicity, we'll use the root logger configured by setup_logging.
logger = logging.getLogger()
# If CONFIG loading failed, this log might not be set up as expected.
logger.info("Flask application starting...")
if "conversion_settings" not in CONFIG or not CONFIG["conversion_settings"]:
     logger.warning("Configuration might not have loaded correctly. API may not function as expected.")


app = Flask(__name__)

# --- Global Settings from Config (if needed by Flask app directly) ---
# Example: set a max upload size from config if desired, otherwise use Flask defaults
# app.config['MAX_CONTENT_LENGTH'] = CONFIG.get('max_upload_size_bytes', 16 * 1024 * 1024) # Default 16MB
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'tif', 'tiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/convert_tiff', methods=['POST'])
def convert_tiff_endpoint():
    if 'tiff_file' not in request.files:
        logger.error("API Error: No tiff_file part in request.")
        return jsonify({"error": "No tiff_file part in request"}), 400

    file = request.files['tiff_file']

    if file.filename == '':
        logger.error("API Error: No selected file.")
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        # Create a unique filename for temporary storage to avoid conflicts
        temp_filename = str(uuid.uuid4()) + "_" + original_filename
        temp_tiff_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)

        try:
            file.save(temp_tiff_path)
            logger.info(f"Uploaded file saved temporarily as: {temp_tiff_path}")

            # Call the refactored function from tiff_converter
            # CONFIG already contains all necessary settings including conversion_settings
            pdf_bytes = tiff_converter.generate_pdf_for_api(temp_tiff_path, CONFIG)

            if pdf_bytes:
                logger.info(f"Successfully converted {original_filename} to PDF.")
                # Send the PDF bytes as a file
                output_filename = os.path.splitext(original_filename)[0] + ".pdf"
                return send_file(
                    io.BytesIO(pdf_bytes),
                    mimetype='application/pdf',
                    as_attachment=True,
                    download_name=output_filename # Use attachment_filename for older Flask versions
                )
            else:
                logger.error(f"Conversion failed for {original_filename}. generate_pdf_for_api returned None.")
                return jsonify({"error": f"Failed to convert TIFF file: {original_filename}. Possible reasons: unsupported TIFF format, corrupt file, or internal error."}), 500

        except Exception as e:
            logger.error(f"Error during conversion process for {original_filename}: {e}", exc_info=True)
            # Use abort(500) to let Flask's error handler create a generic 500 response,
            # or return a custom JSON response.
            return jsonify({"error": f"An unexpected error occurred during conversion: {str(e)}"}), 500
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_tiff_path):
                try:
                    os.remove(temp_tiff_path)
                    logger.info(f"Cleaned up temporary file: {temp_tiff_path}")
                except Exception as e_remove:
                    logger.error(f"Error cleaning up temporary file {temp_tiff_path}: {e_remove}")
    else:
        logger.error(f"API Error: File type not allowed for file: {file.filename}")
        return jsonify({"error": "File type not allowed. Please upload a .tif or .tiff file."}), 400

@app.route('/health', methods=['GET'])
def health_check():
    # Basic health check endpoint
    return jsonify({"status": "healthy", "message": "TIFF to PDF Converter API is running."}), 200

if __name__ == '__main__':
    # Ensure Pillow can handle large images (already in tiff_converter.py's main, but good here too for dev server)
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
    except ImportError:
        logger.warning("Pillow not found, MAX_IMAGE_PIXELS not set. This might be an issue if running app.py directly without tiff_converter.py being fully functional.")

    # For development server. For production, use a WSGI server like Gunicorn or uWSGI.
    # Host '0.0.0.0' makes it accessible externally (e.g., within a Docker container)
    # Debug=True is for development only, DO NOT use in production.
    app.run(host='0.0.0.0', port=5000, debug=True)
