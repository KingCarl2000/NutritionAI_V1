import os
import time
from typing import Any, Literal, cast
from sqlalchemy import text
import mlflow

# Import custom logger
from src.nutrition_core.logging.logger import logger, monitor_performance
from src.postgres.core.connection import get_connection

class PostgresBulkLoader:
    def __init__(self, connection=None):
        """ Nhận connection trực tiếp từ ConnectionPool thay vì tự kết nối. """
        self.conn = connection

    # Áp dụng decorator giám sát hiệu năng (Cảnh báo nếu ETL chạy quá 10 giây)
    @monitor_performance(time_limit=10.0)
    def execute_bulk_load(
        self, 
        table_name: str, 
        file_path_on_server: str,
        schema_name: str = "raw",  # [Cập nhật 1] Hỗ trợ khai báo schema rõ ràng, mặc định đưa vào 'raw'
        delimiter: str = ",",
        format_type: Literal["CSV", "TEXT"] = "CSV",
        has_header: bool = True
    ):
        """
        Nạp dữ liệu từ file vào PostgreSQL bằng lệnh COPY.
        Quản lý Transaction (Commit/Rollback) và MLflow Tracking.
        """
        # Sử dụng connection sẵn có hoặc lấy một cái mới từ pool
        connection = self.conn if self.conn else get_connection()
        
        # Tiền xử lý các thông số tracking
        start_time = time.time()
        file_size_bytes = os.path.getsize(file_path_on_server) if os.path.exists(file_path_on_server) else 0
        row_count = 0

        # Chuẩn bị query COPY
        header_sql = "HEADER" if has_header and format_type == "CSV" else ""
        full_table_name = f'"{schema_name}"."{table_name}"'
        
        copy_sql = f"""
            COPY {full_table_name}
            FROM '{file_path_on_server}'
            WITH (FORMAT {format_type}, DELIMITER '{delimiter}', {header_sql});
        """
        
        logger.info(f"Bắt đầu nạp dữ liệu từ [{file_path_on_server}] vào bảng [{full_table_name}]...")

        try:
            # Lấy Raw DBAPI connection từ SQLAlchemy Connection để dùng .cursor()
            # Tránh lỗi Pylance: Cannot access attribute "cursor"
            raw_conn = cast(
                Any,
                connection.connection if hasattr(connection, "connection") else connection,
            )
            
            # Khởi tạo cursor từ raw connection
            cur = raw_conn.cursor()
            try:
                cur.execute(copy_sql)
                row_count = cur.rowcount
            finally:
                cur.close()

            # SỬA LẠI PHẦN NÀY: Thực hiện commit trực tiếp trên raw_conn
            if hasattr(raw_conn, "commit"):
                raw_conn.commit()
            
            logger.info(f"Đã commit thành công {row_count} dòng.")
            # ... (phần dưới giữ nguyên)

            execution_time = time.time() - start_time
            logger.info(f"Thành công! Đã nạp {row_count} dòng vào {full_table_name} trong {execution_time:.2f} giây.")

            # [MLflow Tracking]
            if mlflow.active_run():
                mlflow.log_param(f"{table_name}_source_path", file_path_on_server)
                mlflow.log_param(f"{table_name}_target_schema", schema_name)
                mlflow.log_metric(f"{table_name}_loaded_rows", row_count)
                mlflow.log_metric(f"{table_name}_file_size_bytes", file_size_bytes)
                mlflow.log_metric(f"{table_name}_etl_duration_sec", execution_time)
            else:
                logger.debug("Không tìm thấy MLflow Active Run để log metrics ETL.")

            return True

        except Exception as e:
            logger.error(f"Lỗi nạp dữ liệu ETL vào bảng {full_table_name}. Đang Rollback... Lỗi: {str(e)}")
            if hasattr(connection, "rollback"):
                connection.rollback()
            elif 'raw_conn' in locals() and hasattr(raw_conn, "rollback"):
                raw_conn.rollback()
            raise e
        finally:
            # Trả connection lại pool nếu class này tự mở
            if not self.conn and hasattr(connection, "close"):
                connection.close()

    def _recreate_indexes_and_fks(self, conn, table_name):
        """Hàm helper: Tạo lại chỉ mục và khóa ngoại. Lưu ý: Tham số giờ là conn thay vì cursor."""
        pass

    # Áp dụng giám sát hiệu năng cho quá trình Analyze (Cảnh báo nếu chạy quá 5 giây)
    @monitor_performance(time_limit=5.0)
    def _run_analyze(self, table_name: str):
        """Chạy ANALYZE cập nhật số liệu thống kê cho planner."""
        logger.info(f"Tiến hành ANALYZE cho bảng {table_name}...")
        
        # Xử lý format schema cho an toàn
        if "." in table_name:
            schema_part, table_part = table_name.split(".", 1)
            safe_table = f'"{schema_part}"."{table_part}"'
        else:
            safe_table = f'"raw"."{table_name}"' # Mặc định rơi vào raw

        with get_connection() as conn:
            # [Cập nhật 3] Transaction handling cho ANALYZE
            try:
                # ANALYZE không chạy được bên trong transaction block ở một số phiên bản Postgres, 
                # do đó cần set isolation_level = AUTOCOMMIT với raw connection hoặc chạy độc lập
                if hasattr(conn, "execution_options"):
                    conn = conn.execution_options(isolation_level="AUTOCOMMIT")
                conn.execute(text(f"ANALYZE {safe_table};"))
                if hasattr(conn, "commit"):
                    conn.commit()
                logger.info(f"Hoàn thành ANALYZE bảng {safe_table}.")
            except Exception as e:
                logger.warning(f"Lỗi khi ANALYZE bảng {safe_table}: {str(e)}")