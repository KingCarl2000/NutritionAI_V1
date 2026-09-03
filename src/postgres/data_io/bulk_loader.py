import psycopg
from psycopg import sql
import logging
from typing import Literal

# Import hàm get_connection từ module core của bạn
from src.postgres.core.connection import get_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostgresBulkLoader:
    def __init__(self):
        """
        Khởi tạo class.
        Không cần nhận connection_params nữa vì chúng ta sử dụng Connection Pool dùng chung.
        """
        pass

    def execute_bulk_load(self, table_name, file_path_on_server, temp_maintenance_work_mem='2GB', format='csv', delimiter=',', header=True):
        r"""
        Thực hiện quá trình ETL nạp dữ liệu lớn tối ưu hiệu năng.
        """
        # Sử dụng 'with' để tự động trả connection về pool sau khi dùng xong
        with get_connection() as conn:
            # autocommit đã được set False từ lúc khởi tạo pool trong connection.py, 
            # nên không cần gán self.conn.autocommit = False nữa.
            
            with conn.cursor() as cursor:
                try:
                    logger.info(f"Bắt đầu quá trình Bulk Load cho bảng '{table_name}' từ file '{file_path_on_server}'...")

                    # 5. Tăng các tham số bộ nhớ đệm của Server (Session level)
                    logger.info(f"Tăng maintenance_work_mem lên {temp_maintenance_work_mem}")
                    cursor.execute(sql.SQL("SET maintenance_work_mem = {};").format(sql.Literal(temp_maintenance_work_mem)))

                    # 3 & 4. Tạm thời gỡ bỏ các chỉ mục (Indexes) và khóa ngoại (Foreign Keys)
                    logger.info("Tạm thời gỡ bỏ các ràng buộc khóa ngoại và chỉ mục (Cần implement chi tiết DDL)...")
                    self._drop_indexes_and_fks(cursor, table_name)

                    # 2. Sử dụng lệnh COPY trực tiếp trên server
                    logger.info("Thực thi lệnh COPY trực tiếp từ server file...")
                    if "." in table_name:
                        schema_part, table_part = table_name.split(".", 1)
                        table_identifier = sql.SQL('.').join([sql.Identifier(schema_part), sql.Identifier(table_part)])
                    else:
                        table_identifier = sql.Identifier(table_name)

                    # Resolve dynamic variables to hardcoded LiteralStrings
                    format_sql = sql.SQL("CSV") if format.lower() == 'csv' else sql.SQL(format) # Add other safe formats if needed
                    header_sql = sql.SQL("TRUE") if header else sql.SQL("FALSE")

                    copy_query = sql.SQL("COPY {} FROM {} WITH (FORMAT {}, DELIMITER {}, HEADER {});").format(
                        table_identifier,
                        sql.Literal(file_path_on_server),
                        format_sql,
                        sql.Literal(delimiter),
                        header_sql
                    )
                    
                    cursor.execute(copy_query)
                    logger.info(f"Đã COPY thành công. Số dòng bị ảnh hưởng: {cursor.rowcount}")

                    # 3 & 4. Xây dựng lại chỉ mục và khóa ngoại sau khi nạp (Rebuild)
                    logger.info("Tái thiết lập các ràng buộc khóa ngoại và xây dựng lại chỉ mục...")
                    self._recreate_indexes_and_fks(cursor, table_name)

                    # Commit giao dịch thông qua biến conn cục bộ
                    conn.commit()
                    logger.info("Transaction COMMIT thành công.")

                except Exception as e:
                    conn.rollback()
                    logger.error(f"Lỗi trong quá trình nạp dữ liệu. Đã ROLLBACK toàn bộ transaction. Lỗi: {e}")
                    raise e
                    
        # 7. Chạy ANALYZE sau khi block with ở trên đã giải phóng xong transaction nạp dữ liệu
        self._run_analyze(table_name)

    def _drop_indexes_and_fks(self, cursor, table_name):
        """Hàm helper: Xóa bỏ chỉ mục và khóa ngoại."""
        pass

    def _recreate_indexes_and_fks(self, cursor, table_name):
        """Hàm helper: Tạo lại chỉ mục và khóa ngoại."""
        pass

    def _run_analyze(self, table_name):
        """Chạy ANALYZE cập nhật số liệu thống kê cho planner."""
        logger.info(f"Tiến hành ANALYZE cho bảng {table_name}...")
        
        if "." in table_name:
            schema_part, table_part = table_name.split(".", 1)
            table_identifier = sql.SQL('.').join([sql.Identifier(schema_part), sql.Identifier(table_part)])
        else:
            table_identifier = sql.Identifier(table_name)

        # Lấy một kết nối mới từ pool để chạy ANALYZE
        with get_connection() as conn:
            # Bật lại autocommit cho kết nối này vì ANALYZE không chạy được trong transaction
            conn.autocommit = True
            with conn.cursor() as cursor:
                try:
                    analyze_query = sql.SQL("ANALYZE {};").format(table_identifier)
                    cursor.execute(analyze_query)
                    logger.info("ANALYZE hoàn tất. Thống kê của planner đã được cập nhật.")
                except Exception as e:
                    logger.error(f"Lỗi khi chạy ANALYZE: {e}")