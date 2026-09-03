import yaml
from pathlib import Path
from src.nutrition_core.logging.logger import logger
from src.postgres.core.connection import get_connection # Import pool connection của bạn

# ... (Giữ nguyên hàm get_pg_data_type)

def create_tables_from_yaml(yaml_path: Path):
    """Đọc file schema.yaml và tạo các bảng tương ứng trong PostgreSQL an toàn."""
    if not yaml_path.exists():
        logger.error(f"❌ Không tìm thấy file schema tại: {yaml_path}")
        return

    # 1. Đọc nội dung file schema.yaml
    with open(yaml_path, 'r', encoding='utf-8') as f:
        schemas = yaml.safe_load(f)
    if not schemas:
        logger.warning(f"File schema trống hoặc không hợp lệ: {yaml_path.name}")
        return

    # 2 & 3. Sử dụng connection pool và transaction để thực thi SQL
    try:
        # Lấy connection từ pool chung
        with get_connection() as conn:
            # psycopg3 cursor được sử dụng để thực thi lệnh
            with conn.cursor() as cursor:
                for dataset_name, tables_dict in schemas.items():
                    logger.info(f"📂 Đang xử lý nhóm dữ liệu: {dataset_name}")
                    
                    if "columns" in tables_dict:
                        tables_dict = {dataset_name: tables_dict}
                        
                    for table_key, table_info in tables_dict.items():
                        schema_name = table_info.get("db_schema", "raw")
                        table_name = table_info.get("table_name", table_key)
                        columns = table_info.get("columns", {})

                        safe_schema = f'"{schema_name}"'
                        safe_table = f'"{table_name}"'
                        full_table_name = f"{safe_schema}.{safe_table}"
                        logger.info(f"Đang xử lý DDL cho bảng {full_table_name}...")

                        # B1: Đảm bảo Schema tồn tại
                        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {safe_schema};")

                        # B2: Xây dựng danh sách các cột an toàn
                        col_definitions = []
                        for col_name, col_attrs in columns.items():
                            pg_type = get_pg_data_type(col_attrs.get("type"))
                            safe_col_name = f'"{col_name}"'
                            col_definitions.append(f"{safe_col_name} {pg_type}")

                        # B3: Xây dựng câu lệnh SQL DDL hoàn chỉnh
                        columns_ddl = ",\n    ".join(col_definitions)
                        drop_sql = f"DROP TABLE IF EXISTS {full_table_name} CASCADE;"
                        create_sql = f"""
                        CREATE TABLE {full_table_name} (
                            {columns_ddl}
                        );
                        """

                        # B4: Thực thi SQL (Xoá bảng cũ và tạo bảng mới)
                        cursor.execute(drop_sql)
                        cursor.execute(create_sql)
                        
                        logger.info(f"✅ Đã tạo bảng {full_table_name} thành công!")
            
            # Khác với SQLAlchemy engine.begin() tự commit, với psycopg pool (autocommit=False)
            # bạn cần gọi commit() thủ công sau khi hoàn tất toàn bộ vòng lặp DDL.
            conn.commit()
            
    except Exception as e:
        # Nếu có lỗi, transaction sẽ tự động bị hủy hoặc bạn có thể gọi rollback rõ ràng
        logger.error(f"❌ Lỗi khi thực thi khởi tạo bảng: {e}")