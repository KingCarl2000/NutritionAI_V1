import psycopg
from psycopg import sql
import logging
from typing import List

logger = logging.getLogger(__name__)

def load_csv_to_postgres(conn: psycopg.Connection, table_name: str, csv_file_path: str, columns: List[str]):
    """
    Sử dụng giao thức COPY của psycopg3 để nạp hàng triệu dòng dữ liệu từ file CSV.
    Quá trình này cực kỳ tối ưu vì bỏ qua các overhead của lệnh INSERT thông thường.
    """
    # Sử dụng psycopg.sql để tạo các Identifier an toàn cho tên bảng và cột
    safe_table = sql.Identifier(table_name)
    safe_columns = sql.SQL(', ').join(map(sql.Identifier, columns))
    
    # Xây dựng câu truy vấn an toàn (trả về kiểu sql.Composed hợp lệ)
    copy_query = sql.SQL("COPY {table} ({fields}) FROM STDIN WITH (FORMAT csv, HEADER true)").format(
        table=safe_table,
        fields=safe_columns
    )
    
    try:
        with conn.cursor() as cur:
            # cur.copy() mở luồng COPY tốc độ cao vào PostgreSQL
            with cur.copy(copy_query) as copy:
                with open(csv_file_path, 'r', encoding='utf-8') as f:
                    # Đọc dòng đầu tiên (Header) để bỏ qua nếu cần, 
                    # nhưng 'HEADER true' trong câu lệnh COPY đã xử lý việc này
                    while data := f.read(8192): # Đọc từng chunk 8KB từ file
                        copy.write(data)
                        
        logger.info(f"Đã nạp xong dữ liệu từ {csv_file_path} vào bảng {table_name} thông qua COPY.")
    except Exception as e:
        logger.error(f"Lỗi khi thực thi COPY: {e}")
        raise