import psycopg
import logging
from psycopg import sql

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CDCStreamManager:
    def __init__(self, primary_conn_info: str):
        """
        Khởi tạo kết nối đến máy chủ cơ sở dữ liệu gốc (Primary/Publisher).
        """
        self.primary_conn_info = primary_conn_info

    def setup_logical_publication(self, pub_name: str, table_name: str):
        """
        Thiết lập Logical Replication với Row Filters và Column Lists.
        Mục đích: Chỉ stream những dữ liệu cần thiết cho Feature Store / MLOps,
        giảm tải cho mạng và tiến trình walsender.
        """
        logger.info(f"Đang thiết lập Publication '{pub_name}' cho bảng '{table_name}'...")
        
        try:
            # Lệnh DDL tạo Publication không thể chạy trong transaction block có chứa nhiều lệnh
            with psycopg.connect(self.primary_conn_info, autocommit=True) as conn:
                with conn.cursor() as cursor:
                # Xóa publication cũ nếu tồn tại
                    drop_query = sql.SQL("DROP PUBLICATION IF EXISTS {};").format(
                        sql.Identifier(pub_name)
                )
                cursor.execute(drop_query)
                
                # TẠO PUBLICATION
                create_query = sql.SQL("""
                    CREATE PUBLICATION {} FOR TABLE {} 
                    (user_id, log_date, weight, calories_burned)
                    WHERE (calories_burned IS NOT NULL AND calories_burned > 0);
                """).format(
                    sql.Identifier(pub_name),
                    sql.Identifier(table_name)
                )
                cursor.execute(create_query)
                    
            logger.info(f"Đã tạo thành công Logical Publication: {pub_name}")
            logger.info("Sẵn sàng stream WAL qua pgoutput tới các hệ thống như RisingWave/TimescaleDB.")
            
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập CDC Publication: {e}")
            raise e

