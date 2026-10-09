from sqlalchemy import create_engine
import psycopg
from psycopg_pool import ConnectionPool

# Import module cấu hình database (lấy URL)
from src.postgres.core import db_config as config

# THAY ĐỔI QUAN TRỌNG: Import custom logger thay vì thư viện logging mặc định
from src.nutrition_core.logging.logger import logger

# Biến global lưu trữ singleton engine
_db_engine = None

def get_engine():
    """
    Khởi tạo và trả về SQLAlchemy Engine theo chuẩn Singleton.
    Sử dụng connection pooling và các tham số timeout an toàn.
    """
    global _db_engine
    
    if _db_engine is None:
        try:
            # Gọi trực tiếp hàm lấy URL chuẩn của SQLAlchemy
            engine_url = config.DatabaseConfig().get_sqlalchemy_url()
            
            # Khởi tạo engine
            _db_engine = create_engine(
                engine_url,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800,
                connect_args={
                    "options": (
                        "-c idle_in_transaction_session_timeout=60000 " # 60 giây ngắt nếu giao dịch treo
                        "-c statement_timeout=30000 "                   # 30 giây tối đa cho truy vấn
                        "-c idle_session_timeout=300000"                # 5 phút ngắt kết nối rảnh
                    )
                }
            )
            # Dòng log này giờ đây sẽ tự động lưu vào thư mục logs/oper_logs/
            logger.info("Đã khởi tạo thành công SQLAlchemy Engine với Connection Pooling.")
            
        except Exception as e:
            # Dòng log này giờ đây sẽ tự động lưu vào thư mục logs/error_logs/
            logger.error(f"Lỗi hệ thống khi khởi tạo SQLAlchemy Engine: {e}")
            raise e
            
    return _db_engine