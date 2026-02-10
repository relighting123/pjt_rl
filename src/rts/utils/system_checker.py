import os
import torch
import psutil
import logging
from ..data.data_loader import DataLoader

logger = logging.getLogger(__name__)

def pre_flight_check(data_dir: str):
    logger.info("Starting pre-flight check...")
    
    # 1. Data Directory Check
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return False
    
    scenarios = DataLoader.list_scenarios(data_dir)
    if not scenarios:
        logger.error(f"No valid scenarios found in {data_dir}")
        return False
    logger.info(f"Found {len(scenarios)} scenarios.")

    # 2. Hardware Check
    cpu_count = psutil.cpu_count()
    mem = psutil.virtual_memory()
    logger.info(f"Hardware: {cpu_count} CPUs, {mem.total / (1024**3):.1f} GB RAM")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    else:
        logger.warning("GPU is not available. Training might be slow.")

    # 3. Disk Space Check
    usage = psutil.disk_usage('.')
    if usage.free < 1024**3: # less than 1GB
        logger.warning(f"Low disk space: {usage.free / (1024**3):.2f} GB available.")
    
    logger.info("Pre-flight check completed successfully.")
    return True

def log_system_status():
    status_logger = logging.getLogger("rts.status")
    cpu_usage = psutil.cpu_percent()
    mem_usage = psutil.virtual_memory().percent
    status_logger.info(f"STATUS - CPU: {cpu_usage}%, MEM: {mem_usage}%")
