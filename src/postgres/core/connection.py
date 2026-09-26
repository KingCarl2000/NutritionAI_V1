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
            # Lấy chuỗi kết nối từ db_config hiện tại
            base_string = config.get_connection_string()
            
            # Chuyển đổi sang định dạng SQLAlchemy hỗ trợ driver psycopg2
            engine_url = base_string.replace("postgresql://", "postgresql+psycopg2://").replace("postgres://", "postgresql+psycopg2://")
            
            # Khởi tạo engine với connection pooling của SQLAlchemy
            _db_engine = create_engine(engine_url, pool_size=5, max_overflow=10)
            logger.info("Đã khởi tạo thành công SQLAlchemy Engine.")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo SQLAlchemy Engine: {e}")
            raise e
            
    return _db_engine
