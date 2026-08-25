import psycopg
from psycopg import sql
import logging
from typing import Iterator, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureStoreEngine:
    def __init__(self, conn_info: str):
        """
        Khởi tạo Feature Store Engine.
        conn_info: Chuỗi kết nối PostgreSQL (vd: "dbname=nutrition_db user=postgres password=secret")
        """
        self.conn_info = conn_info

    def copy_from_csv(self, table_name: str, file_path: str):
        """
        Nạp dữ liệu trực tiếp từ file CSV phía Client lên máy chủ bằng COPY.
        Giúp tránh nghẽn I/O khi cần khởi tạo Feature Store từ file thô.
        """
        logger.info(f"Đang nạp file {file_path} vào bảng Feature Store: {table_name}...")
        
        with psycopg.connect(self.conn_info) as conn:
            with conn.cursor() as cursor:
                # Sử dụng COPY FROM STDIN để stream file từ máy khách lên server
                copy_query = sql.SQL(
                    "COPY {} FROM STDIN WITH (FORMAT CSV, HEADER true)"
                ).format(sql.Identifier(table_name))
                
                with cursor.copy(copy_query) as copy_op:
                    with open(file_path, 'rb') as f:
                        # Ghi dữ liệu file thẳng vào stream của PostgreSQL
                        copy_op.write(f.read())
                        
            conn.commit()
            
        logger.info("Quá trình COPY file hoàn tất!")

    def stream_features_from_memory(self, table_name: str, columns: Tuple[str, ...], data_stream: Iterator[Tuple]):
        """
        Stream dữ liệu từ Python (Generator/Iterator) vào DB thông qua giao thức COPY.
        Phù hợp khi kết quả Feature Engineering đang nằm trên RAM (ví dụ từ Pandas DataFrame 
        trong các pipeline MLOps) và cần đẩy nhanh vào DB mà không dùng INSERT từng dòng.
        """
        col_names = ", ".join(columns)
        logger.info(f"Đang stream dữ liệu in-memory vào bảng {table_name} ({col_names})...")
        
        with psycopg.connect(self.conn_info) as conn:
            with conn.cursor() as cursor:
                copy_query = sql.SQL("COPY {} ({}) FROM STDIN").format(
                    sql.Identifier(table_name),
                    sql.SQL(", ").join(sql.Identifier(column) for column in columns),
                )
                
                with cursor.copy(copy_query) as copy_op:
                    # Ghi từng dòng dữ liệu vào stream
                    for row in data_stream:
                        copy_op.write_row(row)
                        
            conn.commit()
            
        logger.info("Quá trình stream dữ liệu in-memory hoàn tất!")

# --- Ví dụ sử dụng Engine ---
if __name__ == "__main__":
    db_url = "postgresql://postgres:password@localhost:5432/nutrition_db"
    engine = FeatureStoreEngine(db_url)
    
    # Cách 1: Nạp batch từ file CSV thô
    # engine.copy_from_csv("features_table", "Data/raw_features.csv")
    
    # Cách 2: Nạp streaming từ Pipeline xử lý in-memory
    def generate_engineered_features():
        # Giả lập dữ liệu đặc trưng đã qua xử lý
        for i in range(50000):
            yield (i, "2024-01-01", 150.5, 500 + i)
            
    engine.stream_features_from_memory(
        table_name="user_workout_summary",
        columns=("user_id", "log_date", "weight", "calories_burned"),
        data_stream=generate_engineered_features()
    )