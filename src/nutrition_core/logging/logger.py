import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from logging import StreamHandler

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

# Lớp bọc StreamHandler để ép encoding là utf-8, tránh lỗi UnicodeEncodeError trên Windows (cp1252)
class Utf8StreamHandler(StreamHandler):
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stdout
        super().__init__(stream)
        
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            if hasattr(stream, "buffer"):
                stream.buffer.write((msg + self.terminator).encode("utf-8", errors="replace"))
                stream.buffer.flush()
            else:
                stream.write(msg + self.terminator)
                stream.flush()
        except Exception:
            self.handleError(record)

# 3. Gắn cả 2 handlers: Ghi ra file VÀ In ra màn hình console bằng UTF-8 handler
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),  # Ghi vào file với UTF-8
        Utf8StreamHandler(sys.stdout)                         # In ra Terminal an toàn với ký tự đặc biệt/emoji
    ]
)

# 4. Export đối tượng logger chung để các file khác có thể import
logger = logging.getLogger("NutritionAI")