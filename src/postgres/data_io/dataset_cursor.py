import psycopg
import logging
from typing import Iterator, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetCursorExtrator:
    def __init__(self, conn_info: str):
        """
        Khởi tạo kết nối với psycopg3.
        conn_info: Chuỗi kết nối PostgreSQL (vd: "dbname=nutrition_db user=postgres password=secret")
        """
        self.conn_info = conn_info

    def fetch_data_in_chunks(self, query: str, chunk_size: int = 10000) -> Iterator[List[Dict[str, Any]]]:
        """
        Trích xuất dữ liệu theo từng phần (chunks) sử dụng Server-side Cursors.
        
        Args:
            query (str): Câu lệnh SELECT cần thực thi.
            chunk_size (int): Số lượng bản ghi tải về trong mỗi chunk.
            
        Yields:
            Iterator[List[Dict]]: Một batch dữ liệu dạng danh sách các dictionary.
        """
        logger.info(f"Bắt đầu stream dữ liệu với chunk_size = {chunk_size}")
        
        # Bắt buộc phải nằm trong một transaction block (autocommit = False) để dùng server-side cursor
        with psycopg.connect(self.conn_info, autocommit=False) as conn:
            # Bật row_factory để trả về dữ liệu dạng dictionary thay vì tuple (tiện lợi cho Pandas/ML)
            conn.row_factory = psycopg.rows.dict_row
            
            # Khởi tạo Server-side Cursor bằng cách đặt tên cho tham số 'name'
            with conn.cursor(name="ml_training_cursor") as cursor:
                # Tính năng itersize trong psycopg3 giúp tự động nạp ngầm từng batch từ server
                cursor.itersize = chunk_size 
                cursor.execute(query)
                
                while True:
                    # Lấy về một lượng bản ghi vừa đủ với bộ nhớ
                    records = cursor.fetchmany(chunk_size)
                    if not records:
                        break
                    
                    yield records
                    
        logger.info("Hoàn tất quá trình trích xuất dữ liệu.")

# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    db_url = "postgresql://postgres:password@localhost:5432/nutrition_db"
    extractor = DatasetCursorExtrator(db_url)
    
    query = "SELECT user_id, log_date, weight, calories_burned FROM fitness_logs;"
    
    # Dữ liệu sẽ được stream qua RAM thay vì load toàn bộ cùng lúc
    for batch in extractor.fetch_data_in_chunks(query, chunk_size=5000):
        # Tại đây: Chuyển batch thành Pandas DataFrame hoặc đưa trực tiếp vào PyTorch/TensorFlow DataLoader
        print(f"Đã xử lý batch gồm {len(batch)} bản ghi.")