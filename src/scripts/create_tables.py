import sys
from pathlib import Path
import yaml
from sqlalchemy import create_engine, text

# Thiết lập đường dẫn Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.postgres.core.db_config import config
from src.nutrition_core.logging.logger import logger

def get_pg_data_type(yaml_type: str) -> str:
    """Ánh xạ kiểu dữ liệu từ định dạng schema.yaml sang PostgreSQL."""
    type_mapping = {
        "integer": "INTEGER",
        "float": "DOUBLE PRECISION",
        "boolean": "BOOLEAN",
        "datetime": "TIMESTAMP",
        "categorical": "TEXT"
    }
    return type_mapping.get(yaml_type, "TEXT") # Mặc định là TEXT nếu không khớp

def create_tables_from_yaml(yaml_path: Path):
    """Đọc file schema.yaml và tạo các bảng tương ứng trong PostgreSQL."""
    
    if not yaml_path.exists():
        logger.error(f"❌ Không tìm thấy file schema tại: {yaml_path}")
        return

    # 1. Đọc nội dung file schema.yaml
    with open(yaml_path, 'r', encoding='utf-8') as f:
        schemas = yaml.safe_load(f)
        
    if not schemas:
        logger.warning("File schema.yaml trống hoặc không hợp lệ.")
        return

    # 2. Khởi tạo kết nối SQLAlchemy
    db_url = f"postgresql+psycopg2://{config.user}:{config.password}@{config.host}:{config.port}/{config.dbname}"
    engine = create_engine(db_url)

    # 3. Sử dụng transaction để thực thi SQL
    try:
        with engine.begin() as conn: # Tự động commit nếu không lỗi
            for table_key, table_info in schemas.items():
                schema_name = table_info.get("db_schema", "raw")
                table_name = table_info.get("table_name", table_key)
                columns = table_info.get("columns", {})
                
                logger.info(f"Đang xử lý DDL cho bảng '{schema_name}.{table_name}'...")
                
                # B1: Đảm bảo Schema (ví dụ 'raw') tồn tại trong DB
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name};"))
                
                # B2: Xây dựng danh sách các cột
                col_definitions = []
                for col_name, col_attrs in columns.items():
                    pg_type = get_pg_data_type(col_attrs.get("type"))
                    
                    # (Tuỳ chọn) Nếu bạn muốn bắt buộc các cột không được Null ngay từ Raw
                    # is_nullable = "" if col_attrs.get("nullable", True) else " NOT NULL"
                    # Lời khuyên: Ở tầng RAW nên cho phép NULL mọi cột để tránh lỗi Bulk Load
                    
                    col_definitions.append(f'"{col_name}" {pg_type}')
                
                # B3: Xây dựng câu lệnh SQL DDL hoàn chỉnh
                columns_ddl = ",\n    ".join(col_definitions)
                
                drop_sql = f"DROP TABLE IF EXISTS {schema_name}.{table_name} CASCADE;"
                create_sql = f"""
                CREATE TABLE {schema_name}.{table_name} (
                    {columns_ddl}
                );
                """
                
                # B4: Thực thi SQL (Xoá bảng cũ và tạo bảng mới)
                conn.execute(text(drop_sql))
                conn.execute(text(create_sql))
                
                logger.info(f"✅ Đã tạo bảng {schema_name}.{table_name} thành công!")
                
    except Exception as e:
        logger.error(f"❌ Lỗi khi thực thi khởi tạo bảng: {e}")

if __name__ == "__main__":
    # Đường dẫn chuẩn trỏ tới file schema.yaml (chỉnh lại cho khớp thư mục của bạn nếu cần)
    schema_file_path = PROJECT_ROOT / "src" / "api" / "data_schema" / "schema.yaml"
    
    # Thực thi tạo toàn bộ các bảng có trong file YAML
    create_tables_from_yaml(schema_file_path)