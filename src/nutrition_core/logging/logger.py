import logging
import os
from datetime import datetime

# Tách riêng ngày và thời gian chi tiết
current_time = datetime.now()
DATE_DIR = current_time.strftime('%m_%d_%Y')      # Ví dụ: 08_20_2026
LOG_FILE = f"{current_time.strftime('%H_%M_%S')}.log" # Ví dụ: 20_57_42.log

# Tạo đường dẫn thư mục: logs/08_20_2026
logs_path = os.path.join(os.getcwd(), "logs", DATE_DIR)
os.makedirs(logs_path, exist_ok=True)

# File log hoàn chỉnh: logs/08_20_2026/20_57_42.log
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)