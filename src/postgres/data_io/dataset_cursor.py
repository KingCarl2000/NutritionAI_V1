import logging
from typing import Any, cast  # <-- FIXED: Imported 'Any' from typing
import psycopg
from psycopg import rows, sql  # <-- FIXED: Imported 'rows' directly from psycopg to prevent attribute access issues
from src.postgres.core.connection import get_connection
from src.postgres.core.db_config import config

logger = logging.getLogger(__name__)

def stream_training_data(query: str):
    """
    Generator yield các batch dữ liệu sử dụng Server-side Cursors.
    Giải quyết bài toán OOM (Out Of Memory) khi train các Deep Learning Models (PyTorch/TensorFlow).
    """
    # Khởi tạo kết nối. Server-side cursors yêu cầu chạy trong một transaction.
    with get_connection() as conn:
        
        # Đặt tên cho cursor (ví dụ: 'ml_training_cursor') để ép psycopg3 
        # tạo ra một Server-side Cursor thay vì Client-side Cursor mặc định.
        with conn.cursor(name="ml_training_cursor", row_factory=rows.dict_row) as cur:
            logger.info(f"Đã khởi tạo Server-side Cursor. Đang chuẩn bị query: {query[:50]}...")
            
            # FIX: Cast the dynamic 'str' query to a LiteralString-compatible type via Any for sql.SQL()
            # WARNING: Ensure the 'query' variable does not contain unsanitized user input!
            cur.execute(sql.SQL(cast(Any, query)))
            
            while True:
                # fetchmany() sẽ chỉ kéo đúng số dòng bằng config.training_chunk_size (vd: 10000 dòng)
                # qua mạng (network) về RAM của Python, dữ liệu còn lại vẫn nằm trên PostgreSQL Server.
                chunk = cur.fetchmany(config.training_chunk_size)
                
                if not chunk:
                    logger.info("Đã stream toàn bộ dữ liệu huấn luyện.")
                    break
                    
                logger.info(f"Đã tải thành công lô dữ liệu (batch): {len(chunk)} dòng.")
                
                # Sử dụng 'yield' để tạo thành Python Generator, 
                # giúp luồng huấn luyện DL tiêu thụ data tuần tự.
                yield chunk