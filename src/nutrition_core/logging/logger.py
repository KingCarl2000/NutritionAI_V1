import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Xác định đường dẫn thư mục logs gốc
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
current_time = datetime.now()

DATE_DIR = current_time.strftime('%m_%d_%Y')
LOG_FILE = f"{current_time.strftime('%H_%M_%S')}.log"

logs_path = os.path.join(PROJECT_ROOT, "logs", DATE_DIR)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Khởi tạo Logger
logger = logging.getLogger("NutritionAI")
logger.setLevel(logging.INFO) # Đảm bảo nhận mức INFO

# Tránh tạo trùng lặp Handlers khi gọi lại
if not logger.handlers:
    # 1. File Handler (Ghi ra file log)
    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 2. Console Handler (In ra Terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[ %(asctime)s ] %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)