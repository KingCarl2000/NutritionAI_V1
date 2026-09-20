

import psycopg
from psycopg import sql
from typing import Literal

# BỎ thư viện logging mặc định và thay bằng custom logger của hệ thống
from src.nutrition_core.logging.logger import logger, monitor_performance
from src.postgres.core.connection import get_connection


class PostgresBulkLoader:
    def __init__(self, connection):
        """
        Nhận connection trực tiếp từ ConnectionPool thay vì tự kết nối.
        """
        self.conn = connection

    # Áp dụng decorator giám sát hiệu năng (Cảnh báo nếu ETL chạy quá 10 giây)
    @monitor_performance(time_limit=10.0)
    def execute_bulk_load(self, table_name: str, file_path_on_server: str,
                          temp_maintenance_work_mem='2GB',
                          format: Literal["csv", "text", "binary"] = "csv",
                          delimiter: str = ',', header: bool = True):
        r"""
        Thực hiện quá trình ETL nạp dữ liệu lớn tối ưu hiệu năng.
        """
        # Sử dụng 'with' để tự động trả connection về pool sau khi dùng xong
        with get_connection() as conn:
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
                    
                    format_sql = sql.SQL(format.upper())
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

    # Áp dụng giám sát hiệu năng cho quá trình Analyze (Cảnh báo nếu chạy quá 5 giây)
    @monitor_performance(time_limit=5.0)
    def _run_analyze(self, table_name):
        """Chạy ANALYZE cập nhật số liệu thống kê cho planner."""
        logger.info(f"Tiến hành ANALYZE cho bảng {table_name}...")
        if "." in table_name:
            schema_part, table_part = table_name.split(".", 1)
            table_identifier = sql.SQL('.').join([sql.Identifier(schema_part), sql.Identifier(table_part)])
        else:
            table_identifier = sql.Identifier(table_name)
            
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("ANALYZE {};").format(table_identifier))
            conn.commit()
            logger.info(f"Hoàn thành ANALYZE bảng {table_name}.")