import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# 1. Xác định đường dẫn gốc của Project một cách tuyệt đối (NutritionAI_V1)
# logger.py nằm ở: src/nutrition_core/logging/logger.py (sâu 3 cấp so với root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Tách riêng ngày và thời gian chi tiết
current_time = datetime.now()
DATE_DIR = current_time.strftime('%m_%d_%Y')      
LOG_FILE = f"{current_time.strftime('%H_%M_%S')}.log" 

# Tạo đường dẫn thư mục: root/logs/08_20_2026
logs_path = os.path.join(PROJECT_ROOT, "logs", DATE_DIR)
os.makedirs(logs_path, exist_ok=True)

# File log hoàn chỉnh
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# 2. Cấu hình định dạng Log
log_format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"

# 3. Gắn cả 2 handlers: Ghi ra file VÀ In ra màn hình console
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),  # Ghi vào file
        logging.StreamHandler(sys.stdout)    # In ra Terminal
    ]
)

# Export một đối tượng logger chung để các file khác có thể import
logger = logging.getLogger("NutritionAI")