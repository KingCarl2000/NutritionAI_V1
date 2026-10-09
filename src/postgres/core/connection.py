import psycopg
from psycopg_pool import ConnectionPool
from src.postgres.core.db_config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo Connection Pool dùng chung cho toàn bộ ứng dụng
# Thiết lập autocommit mặc định là False để kiểm soát Transaction thủ công trong ETL
try:
    db_pool = ConnectionPool(
        config.get_connection_string(),
        min_size=2,
        max_size=10,
        kwargs={"autocommit": False} 
    )
    logger.info("Đã khởi tạo thành công PostgreSQL Connection Pool (psycopg3).")
except Exception as e:
    logger.error(f"Lỗi khởi tạo Connection Pool: {e}")
    raise

def get_connection():
    """Hàm hỗ trợ lấy connection từ pool."""
    return db_pool.connection()


from sqlalchemy import create_engine

# Biến toàn cục để lưu trữ engine (chỉ tạo 1 lần - Singleton pattern)
_db_engine = None

def get_engine():
    """
    Trả về SQLAlchemy Engine dùng cho Pandas to_sql hoặc ORM.
    """
    global _db_engine
    if _db_engine is None:
        try:
            # Gọi trực tiếp hàm lấy URL chuẩn của SQLAlchemy
            engine_url = config.get_sqlalchemy_url()
            
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
                        "-c statement_timeout=30000 "                   # 30 giây tối đa cho một truy vấn (tránh query nặng)
                        "-c idle_session_timeout=300000"                # 5 phút ngắt kết nối rảnh rỗi
            )
                } 
            logger.info("Đã khởi tạo thành công SQLAlchemy Engine.")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo SQLAlchemy Engine: {e}")
            raise e
            
    return _db_engine
