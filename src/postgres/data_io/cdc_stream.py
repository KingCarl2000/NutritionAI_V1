import logging
from typing import List
import psycopg
from psycopg import sql
from psycopg.errors import DuplicateObject

from src.postgres.core.connection import get_connection
from src.postgres.core.db_config import config
from src.postgres.data_io.bulk_loader import load_csv_to_postgres

logger = logging.getLogger(__name__)

def run_optimized_bulk_load(csv_path: str, table_name: str, columns: List[str]):
    """
    Quản lý toàn bộ vòng đời ETL:
    1. Disable Autocommit (Mặc định trong connection pool của chúng ta)
    2. Tăng maintenance_work_mem
    3. Drop Index (Giả lập)
    4. COPY Data
    5. Recreate Index (Giả lập)
    6. Chạy ANALYZE
    7. Commit (hoặc Rollback nếu lỗi)
    """
    
    with get_connection() as conn:
        try:
            with conn.cursor() as cur:
                logger.info("--- BẮT ĐẦU TRANSACTION ETL ---")
                
                # Tăng bộ nhớ tạm thời cho worker
                # Assuming config.maintenance_work_mem is a safe string, but still good practice to be careful.
                cur.execute(sql.SQL("SET maintenance_work_mem = {};").format(sql.Literal(config.maintenance_work_mem)))
                
                # BƯỚC 1: Drop Index & Ràng buộc
                logger.info("Tạm thời vô hiệu hóa Index và Ràng buộc...")
                index_name = f"idx_{table_name}_user_time"
                cur.execute(
                    sql.SQL("DROP INDEX IF EXISTS {};").format(sql.Identifier(index_name))
                )
                
                # BƯỚC 2: Thực thi giao thức COPY
                load_csv_to_postgres(conn, table_name, csv_path, columns)
                
                # BƯỚC 3: Rebuild Index
                logger.info("Đang tạo lại (Rebuild) Indexes...")
                # Note: In a real scenario, you'd likely want the index columns to be dynamic as well, 
                # but I'm keeping the original logic of indexing user_id and meal_time here.
                cur.execute(
                    sql.SQL("CREATE INDEX {} ON {} (user_id, meal_time);").format(
                        sql.Identifier(index_name),
                        sql.Identifier(table_name)
                    )
                )
                
                # Commit dữ liệu và schema changes
                conn.commit() 
                logger.info("Đã COMMIT Transaction. Dữ liệu nạp thành công.")
                
            # ANALYZE không thể chạy trong transaction block có chứa lỗi trước đó, 
            # tốt nhất nên chạy trên một kết nối autocommit mới
            conn.autocommit = True
            with conn.cursor() as cur:
                logger.info("Đang chạy lệnh ANALYZE cập nhật bộ lập lịch (planner)...")
                cur.execute(
                    sql.SQL("ANALYZE {};").format(sql.Identifier(table_name))
                )
                
        except Exception as e:
            conn.rollback() # Khôi phục lại trạng thái ban đầu nếu có bất kỳ lỗi nào
            logger.error(f"Gặp lỗi. Đã ROLLBACK toàn bộ Transaction. Lỗi: {e}")

def _ensure_replication_slot_exists(conn, slot_name: str, plugin: str = "test_decoding"):
    """Helper function to check and create a replication slot if it doesn't exist."""
    with conn.cursor() as cur:
        # Check if the slot exists
        cur.execute(
            "SELECT 1 FROM pg_replication_slots WHERE slot_name = %s",
            (slot_name,)
        )
        if not cur.fetchone():
            logger.info(f"Replication slot '{slot_name}' not found. Creating it...")
            try:
                # Sử dụng sql.SQL để Pylance không báo lỗi LiteralString và an toàn hơn
                query = sql.SQL("CREATE_REPLICATION_SLOT {} LOGICAL {};").format(
                    sql.Identifier(slot_name),
                    sql.Identifier(plugin)
                )
                cur.execute(query)
            except DuplicateObject:
                # Fallback in case of a race condition
                logger.warning(f"Replication Slot '{slot_name}' already exists.")
        else:
            logger.info(f"Replication slot '{slot_name}' already exists.")

def listen_to_wal_stream(slot_name: str = "ml_cdc_slot"):
    """
    Sử dụng psycopg3 để lắng nghe (stream) Logical Replication từ Write-Ahead Log (WAL).
    Yêu cầu cấu hình DB: wal_level = logical.
    Phục vụ cho Real-time Inference khi người dùng vừa nhập món ăn mới lên App.
    """
    conn_str = config.get_connection_string()
    
    try:
        # Require a dedicated connection with replication='database'
        with psycopg.connect(conn_str, autocommit=True, replication="database") as conn:
            
            # 1. Ensure the replication slot exists safely
            _ensure_replication_slot_exists(conn, slot_name)
            
            # 2. Start streaming
        with conn.cursor() as cur:
            logger.info("Đang bắt đầu stream từ WAL log...")
            
            # Sử dụng sql.SQL thay vì f-string 
            query = sql.SQL("START_REPLICATION SLOT {} LOGICAL 0/0;").format(
                sql.Identifier(slot_name)
            )
            cur.execute(query)
            
            # Lặp vô hạn để nhận các thay đổi
            for message in cur:
                    payload = message.payload.decode('utf-8')
                    
                    if "INSERT" in payload:
                        logger.info(f"[CDC EVENT] Nhận dữ liệu mới: {payload}")
                        
                        # TODO: Gửi payload này sang Message Queue
                        
                        # Gửi tín hiệu báo cho Postgres biết đã xử lý xong LSN này
                        # This requires psycopg.replication module for proper handling in a real app
                        # message.cursor.send_feedback(flush_lsn=message.data_start)
                        
    except Exception as e:
        logger.error(f"Lỗi Logical Replication CDC: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage:
    # run_optimized_bulk_load(
    #     csv_path="data.csv", 
    #     table_name="nutrition_logs", 
    #     columns=["user_id", "meal_time", "food_item", "calories", "protein", "fat", "carbs"]
    # )
    # listen_to_wal_stream()