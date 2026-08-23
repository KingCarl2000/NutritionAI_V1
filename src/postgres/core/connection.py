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