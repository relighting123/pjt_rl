import logging
import sys
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir="logs", level=logging.INFO):
    # Prevent duplicate handlers if called multiple times
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        return

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(level)

    # Main Log File
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "rts.log"),
        maxBytes=10*1024*1024, # 10MB
        backupCount=5
    )
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.DEBUG)

    # Error Log File
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, "error.log"),
        maxBytes=10*1024*1024,
        backupCount=5
    )
    error_handler.setFormatter(log_format)
    error_handler.setLevel(logging.ERROR)

    # Status/Heartbeat Log File
    status_handler = RotatingFileHandler(
        os.path.join(log_dir, "system_status.log"),
        maxBytes=5*1024*1024,
        backupCount=3
    )
    status_handler.setFormatter(log_format)
    status_handler.setLevel(logging.INFO)

    # Setup Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)

    # Status logger for heartbeat
    status_logger = logging.getLogger("rts.status")
    status_logger.propagate = False
    status_logger.addHandler(status_handler)
    status_logger.addHandler(console_handler) # also show in console

    logger = logging.getLogger(__name__)
    logger.info("Logging initialized.")

def get_status_logger():
    return logging.getLogger("rts.status")

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception
