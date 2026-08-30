import psycopg2
from psycopg2 import sql
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostgresBulkLoader:
    def __init__(self, connection_params):
        """Khởi tạo kết nối đến database."""
        self.conn = psycopg2.connect(**connection_params)
        
    def execute_bulk_load(self, table_name, file_path_on_server, temp_maintenance_work_mem='2GB', format='csv', delimiter=',', header=True):
        r"""
        Thực hiện quá trình ETL nạp dữ liệu lớn tối ưu hiệu năng. Có thể cấu hình cho nhiều loại file.
        
        Args:
            table_name (str): Tên bảng cần nạp dữ liệu.
            file_path_on_server (str): Đường dẫn tuyệt đối đến file nằm trên SERVER chạy PostgreSQL.
            temp_maintenance_work_mem (str): Dung lượng RAM cấp thêm tạm thời cho maintenance.
            format (str): Định dạng file, ví dụ: 'csv' hoặc 'text'. Mặc định 'csv'.
            delimiter (str): Ký tự phân cách cột. Mặc định là ',' cho csv. Nếu là tsv có thể dùng '\t'.
            header (bool): Bỏ qua dòng tiêu đề hay không. Mặc định là True (có bỏ qua).
        """
        # 1. Tắt chế độ tự động commit (Disable Autocommit)
        # gom tất cả vào một transaction duy nhất để đảm bảo an toàn và tối ưu ghi đĩa.
        self.conn.autocommit = False
        cursor = self.conn.cursor()

        try:
            logger.info(f"Bắt đầu quá trình Bulk Load cho bảng '{table_name}' từ file '{file_path_on_server}'...")
            
            # 5. Tăng các tham số bộ nhớ đệm của Server (Session level)
            logger.info(f"Tăng maintenance_work_mem lên {temp_maintenance_work_mem}")
            cursor.execute(sql.SQL("SET maintenance_work_mem = {};").format(sql.Literal(temp_maintenance_work_mem)))
            
            # 3 & 4. Tạm thời gỡ bỏ các chỉ mục (Indexes) và khóa ngoại (Foreign Keys)
            logger.info("Tạm thời gỡ bỏ các ràng buộc khóa ngoại và chỉ mục (Cần implement chi tiết DDL)...")
            self._drop_indexes_and_fks(cursor, table_name)

            # 2. Sử dụng lệnh COPY
            # Dùng đường dẫn file TRỰC TIẾP TRÊN SERVER để tiến trình Postgres tự đọc
            logger.info(f"Thực thi lệnh COPY trực tiếp từ server file...")
            
            # Xử lý trường hợp table_name chứa schema (VD: "raw.apple_data_raw")
            if "." in table_name:
                schema_part, table_part = table_name.split(".", 1)
                table_identifier = sql.SQL('.').join([sql.Identifier(schema_part), sql.Identifier(table_part)])
            else:
                table_identifier = sql.Identifier(table_name)

            # Xây dựng câu lệnh COPY động dựa trên tham số
            copy_query = sql.SQL("COPY {} FROM {} WITH (FORMAT {}, DELIMITER {}, HEADER {});").format(
                table_identifier,
                sql.Literal(file_path_on_server),
                sql.SQL(format),
                sql.Literal(delimiter),
                sql.SQL(str(header).upper())
            )
            
            cursor.execute(copy_query)
            logger.info(f"Đã COPY thành công. Số dòng bị ảnh hưởng: {cursor.rowcount}")

            # 3 & 4. Xây dựng lại chỉ mục và khóa ngoại sau khi nạp (Rebuild)
            logger.info("Tái thiết lập các ràng buộc khóa ngoại và xây dựng lại chỉ mục...")
            self._recreate_indexes_and_fks(cursor, table_name)

            # Commit giao dịch
            self.conn.commit()
            logger.info("Transaction COMMIT thành công.")

        except Exception as e:
            # Rollback nếu có bất kỳ lỗi nào xảy ra trong quá trình COPY
            self.conn.rollback()
            logger.error(f"Lỗi trong quá trình nạp dữ liệu. Đã ROLLBACK toàn bộ transaction. Lỗi: {e}")
            raise e
        finally:
            cursor.close()

        # 7. Chạy ANALYZE sau khi hoàn tất nạp dữ liệu
        self._run_analyze(table_name)

    def _drop_indexes_and_fks(self, cursor, table_name):
        """
        Hàm helper: Xóa bỏ chỉ mục và khóa ngoại.
        (Cần thay thế bằng các câu lệnh ALTER TABLE ... DROP CONSTRAINT và DROP INDEX thực tế của bạn).
        """
        # TODO: Cập nhật logic xóa index/FK động theo table_name nếu cần
        pass

    def _recreate_indexes_and_fks(self, cursor, table_name):
        """
        Hàm helper: Tạo lại chỉ mục và khóa ngoại.
        (Cần thay thế bằng các câu lệnh ALTER TABLE ... ADD CONSTRAINT và CREATE INDEX thực tế của bạn).
        """
        # TODO: Cập nhật logic tạo lại index/FK động theo table_name nếu cần
        pass

    def _run_analyze(self, table_name):
        """Chạy ANALYZE cập nhật số liệu thống kê cho planner."""
        logger.info(f"Tiến hành ANALYZE cho bảng {table_name}...")

        # Xử lý trường hợp table_name chứa schema
        if "." in table_name:
            schema_part, table_part = table_name.split(".", 1)
            table_identifier = sql.SQL('.').join([sql.Identifier(schema_part), sql.Identifier(table_part)])
        else:
            table_identifier = sql.Identifier(table_name)

        # Bật lại autocommit vì ANALYZE không chạy được trong transaction (BEGIN ... COMMIT)
        self.conn.autocommit = True
        cursor = self.conn.cursor()

        try:
            analyze_query = sql.SQL("ANALYZE {};").format(table_identifier)
            cursor.execute(analyze_query)
            logger.info("ANALYZE hoàn tất. Thống kê của planner đã được cập nhật.")
        except Exception as e:
            logger.error(f"Lỗi khi chạy ANALYZE: {e}")
        finally:
            cursor.close()
            self.conn.autocommit = False

