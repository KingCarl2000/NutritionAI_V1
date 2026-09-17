import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from functools import wraps

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None

# Xác định đường dẫn thư mục logs gốc
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
current_time = datetime.now()

DATE_DIR = current_time.strftime('%m_%d_%Y')
LOG_FILE = f"{current_time.strftime('%H_%M_%S')}.log"

# --- TẠO 3 THƯ MỤC RIÊNG BIỆT ---
oper_logs_path = os.path.join(PROJECT_ROOT, "logs", "oper_logs", DATE_DIR)
warm_logs_path = os.path.join(PROJECT_ROOT, "logs", "warm_logs", DATE_DIR)
error_logs_path = os.path.join(PROJECT_ROOT, "logs", "error_logs", DATE_DIR)

os.makedirs(oper_logs_path, exist_ok=True)
os.makedirs(warm_logs_path, exist_ok=True)
os.makedirs(error_logs_path, exist_ok=True)

OPER_LOG_FILE_PATH = os.path.join(oper_logs_path, LOG_FILE)
WARM_LOG_FILE_PATH = os.path.join(warm_logs_path, LOG_FILE)
ERROR_LOG_FILE_PATH = os.path.join(error_logs_path, LOG_FILE)

# Khởi tạo Logger
logger = logging.getLogger("NutritionAI")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")

    # 1. File Handler cho oper_logs (Lưu tất cả mọi thứ từ INFO)
    oper_handler = logging.FileHandler(OPER_LOG_FILE_PATH, encoding='utf-8')
    oper_handler.setLevel(logging.INFO)
    oper_handler.setFormatter(formatter)
    logger.addHandler(oper_handler)

    # 2. File Handler cho warm_logs (Lưu từ WARNING trở lên)
    warm_handler = logging.FileHandler(WARM_LOG_FILE_PATH, encoding='utf-8')
    warm_handler.setLevel(logging.WARNING) 
    warm_handler.setFormatter(formatter)
    logger.addHandler(warm_handler)

    # 3. File Handler cho error_logs (Chỉ lưu ERROR và CRITICAL)
    error_handler = logging.FileHandler(ERROR_LOG_FILE_PATH, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    # 4. Console Handler (In ra Terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[ %(asctime)s ] %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

# ==========================================
# CÁC HÀM CẢNH BÁO HIỆU NĂNG VÀ PHẦN CỨNG
# ==========================================

def monitor_performance(time_limit=5.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            exec_time = time.time() - start_time
            
            if exec_time > time_limit:
                # Ghi vào oper_logs và warm_logs
                logger.warning(
                    f"[HIỆU NĂNG] Hàm '{func.__name__}' chạy quá chậm: {exec_time:.2f}s "
                    f"(Ngưỡng cảnh báo: {time_limit}s)."
                )
            else:
                # Chỉ ghi vào oper_logs
                logger.info(f"Hàm '{func.__name__}' hoàn thành trong {exec_time:.2f}s.")
            return result
        return wrapper
    return decorator


def check_hardware_limits():
    if psutil:
        cpu_usage = psutil.cpu_percent(interval=0.5)
        if cpu_usage > 90.0:
            logger.warning(f"[HARDWARE] CPU (i7) đang quá tải: {cpu_usage}%!")
        
        ram = psutil.virtual_memory()
        if ram.percent > 85.0:
            used_gb = ram.used / (1024 ** 3)
            logger.warning(f"[HARDWARE] RAM sắp hết: {ram.percent}% ({used_gb:.1f}GB / 32.0GB đã dùng)!")
    else:
        # Lỗi này sẽ lưu vào CẢ 3 THƯ MỤC: oper_logs, warm_logs, và error_logs
        logger.error("Thư viện 'psutil' chưa được cài đặt. Chạy: pip install psutil")

    if pynvml:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            gpu_used_gb = info.used / (1024 ** 3)
            gpu_total_gb = info.total / (1024 ** 3)
            gpu_percent = (info.used / info.total) * 100
            
            if gpu_percent > 85.0:
                logger.warning(
                    f"[HARDWARE] GPU (A3000) VRAM đang cạn kiệt: {gpu_percent:.1f}% "
                    f"({gpu_used_gb:.2f}GB / {gpu_total_gb:.2f}GB)!"
                )
        except pynvml.NVMLError as e:
            # Lỗi này sẽ lưu vào CẢ 3 THƯ MỤC
<<<<<<< HEAD
            logger.error(f"Không thể đọc thông số GPU: {e}")
=======
            logger.error(f"Không thể đọc thông số GPU: {e}")
>>>>>>> c5bbc66907343c81c99e1ca3a7adeccf86d9b473
