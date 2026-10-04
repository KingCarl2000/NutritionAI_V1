import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from functools import wraps
import dvc.exceptions as dvc_exc
import subprocess


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
# CÁC HÀM CẢNH BÁO HIỆU NĂNG THỜI GIAN
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
                logger.warning(f"[HIỆU NĂNG] Hàm '{func.__name__}' chạy quá chậm: {exec_time:.2f}s "
                               f"(Ngưỡng cảnh báo: {time_limit}s)."
                               )
            else:
                # Chỉ ghi vào oper_logs
                logger.info(f"Hàm '{func.__name__}' hoàn thành trong {exec_time:.2f}s.")
            return result
        return wrapper
    return decorator

# Thêm vào cuối file src/nutrition_core/logging/logger.py

def dvc_error_handler(max_retries: int = 3, retry_delay: int = 2, fallback_func=None):
    """
    Decorator xử lý lỗi tập trung cho DVC: Map ngoại lệ, Inject Context, và Fallback & Retry.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Import cục bộ để tránh Circular Import với exception.py
            from src.nutrition_core.exception.exception import (
                DataPipelineException, 
                DataNotFoundError, 
                RemoteStorageError,
                VersionNotFoundError
            )
            
            retries = 0
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                    
                except (dvc_exc.PathMissingError, dvc_exc.FileMissingError, FileNotFoundError) as e:
                    # 1 & 2: Map Exception & Inject Context (Ghi log tham số đầu vào)
                    logger.error(f"[DVC NOT FOUND] Không tìm thấy dữ liệu trong hàm '{func.__name__}'. "
                                 f"Context: args={args}, kwargs={kwargs}. Chi tiết: {e}")
                    raise DataNotFoundError(f"Dữ liệu hoặc mô hình không tồn tại: {e}")
                    
                except dvc_exc.RevisionError as e:
                    logger.error(f"[DVC VERSION WARN] Sai revision/commit trong '{func.__name__}': {e}")
                    raise VersionNotFoundError(f"Không tìm thấy phiên bản dữ liệu: {e}")
                    
                except (subprocess.CalledProcessError, Exception) as e:
                    # Kiểm tra xem có phải lỗi mạng / Remote Storage không (dựa vào thông báo lỗi)
                    error_msg = str(e).lower()
                    is_network_error = any(kw in error_msg for kw in ["network", "connection", "remote", "credentials", "s3", "gcs", "timeout"])
                    
                    if is_network_error or isinstance(e, subprocess.CalledProcessError):
                        logger.warning(f"[DVC NETWORK WARN] Lỗi giao tiếp Remote trong '{func.__name__}'. "
                                       f"Lần thử {retries}/{max_retries}. Chi tiết: {e}")
                        
                        # 3. Retry
                        if retries < max_retries:
                            retries += 1
                            time.sleep(retry_delay)
                            continue
                            
                        # 3. Fallback (Cơ chế dự phòng)
                        if fallback_func:
                            logger.info(f"[DVC FALLBACK] Đang kích hoạt cơ chế dự phòng cục bộ cho '{func.__name__}'...")
                            try:
                                return fallback_func(*args, **kwargs)
                            except Exception as fallback_err:
                                logger.error(f"[DVC FALLBACK FAILED] Cơ chế dự phòng thất bại: {fallback_err}")
                        
                        logger.error(f"[DVC ERROR] Mất kết nối Remote Storage hoàn toàn sau {max_retries} lần thử.")
                        raise RemoteStorageError(f"Không thể kết nối DVC Remote: {e}")
                    
                    # Các lỗi hệ thống khác không liên quan đến mạng
                    logger.error(f"[DVC SYSTEM ERROR] Lỗi không xác định trong '{func.__name__}': {e}")
                    raise DataPipelineException(f"Lỗi DVC nội bộ: {e}")
        return wrapper
    return decorator