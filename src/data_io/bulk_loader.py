from typing import Literal
from sqlalchemy import text

# Import custom logger
from src.nutrition_core.logging.logger import logger, monitor_performance
from src.postgres.core.connection import get_connection

class PostgresBulkLoader:
    def __init__(self, connection=None):
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
        Sử dụng cú pháp chuẩn SQLAlchemy.
        """
        with get_connection() as conn:
            try:
                logger.info(f"Bắt đầu quá trình Bulk Load cho bảng '{table_name}' từ file '{file_path_on_server}'...")

                # 5. Tăng các tham số bộ nhớ đệm của Server (Session level)
                logger.info(f"Tăng maintenance_work_mem lên {temp_maintenance_work_mem}")
                conn.execute(text(f"SET maintenance_work_mem = '{temp_maintenance_work_mem}';"))

                # 3 & 4. Tạm thời gỡ bỏ các chỉ mục (Indexes) và khóa ngoại (Foreign Keys)
                logger.info("Tạm thời gỡ bỏ các ràng buộc khóa ngoại và chỉ mục (Cần implement chi tiết DDL)...")
                self._drop_indexes_and_fks(conn, table_name)

                # 2. Sử dụng lệnh COPY trực tiếp trên server
                logger.info("Thực thi lệnh COPY trực tiếp từ server file...")
                
                # Xử lý an toàn tên bảng / schema
                if "." in table_name:
                    schema_part, table_part = table_name.split(".", 1)
                    safe_table = f'"{schema_part}"."{table_part}"'
                else:
                    safe_table = f'"{table_name}"'

                header_str = "TRUE" if header else "FALSE"
                
                # Cấu trúc câu lệnh COPY chuẩn
                copy_query = f"""
                    COPY {safe_table} 
                    FROM '{file_path_on_server}' 
                    WITH (FORMAT {format.upper()}, DELIMITER '{delimiter}', HEADER {header_str});
                """
                
                # SQLAlchemy execute trả về CursorResult, hỗ trợ thuộc tính rowcount
                result = conn.execute(text(copy_query))
                logger.info(f"Đã COPY thành công. Số dòng bị ảnh hưởng: {result.rowcount}")

                # 3 & 4. Xây dựng lại chỉ mục và khóa ngoại sau khi nạp (Rebuild)
                logger.info("Tái thiết lập các ràng buộc khóa ngoại và xây dựng lại chỉ mục...")
                self._recreate_indexes_and_fks(conn, table_name)

                # Commit giao dịch
                conn.commit()
                logger.info("Transaction COMMIT thành công.")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Lỗi trong quá trình nạp dữ liệu. Đã ROLLBACK toàn bộ transaction. Lỗi: {e}")
                raise e

        # 7. Chạy ANALYZE sau khi block with ở trên đã giải phóng xong transaction nạp dữ liệu
        self._run_analyze(table_name)

    def _drop_indexes_and_fks(self, conn, table_name):
        """Hàm helper: Xóa bỏ chỉ mục và khóa ngoại. Lưu ý: Tham số giờ là conn thay vì cursor."""
        pass

    def _recreate_indexes_and_fks(self, conn, table_name):
        """Hàm helper: Tạo lại chỉ mục và khóa ngoại. Lưu ý: Tham số giờ là conn thay vì cursor."""
        pass

    # Áp dụng giám sát hiệu năng cho quá trình Analyze (Cảnh báo nếu chạy quá 5 giây)
    @monitor_performance(time_limit=5.0)
    def _run_analyze(self, table_name):
        """Chạy ANALYZE cập nhật số liệu thống kê cho planner."""
        logger.info(f"Tiến hành ANALYZE cho bảng {table_name}...")
        
        if "." in table_name:
            schema_part, table_part = table_name.split(".", 1)
            safe_table = f'"{schema_part}"."{table_part}"'
        else:
            safe_table = f'"{table_name}"'

        with get_connection() as conn:
            conn.execute(text(f"ANALYZE {safe_table};"))
            conn.commit()
            
        logger.info(f"Hoàn thành ANALYZE bảng {table_name}.")