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
        "integer": "BIGINT",           # BẮT BUỘC: Dùng BIGINT để tránh lỗi tràn số với Id thiết bị (VD: 1503960366)
        "float": "DOUBLE PRECISION",
        "boolean": "BOOLEAN",
        "datetime": "TIMESTAMPTZ",     # CẢI TIẾN: Luôn dùng TIMESTAMP WITH TIME ZONE cho dữ liệu Health/Activity
        "categorical": "TEXT"
    }
    return type_mapping.get(yaml_type, "TEXT") # Mặc định là TEXT nếu không khớp

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

    # 2. Khởi tạo kết nối SQLAlchemy
    db_url = f"postgresql+psycopg2://{config.user}:{config.password}@{config.host}:{config.port}/{config.dbname}"
    engine = create_engine(db_url)

    # 3. Sử dụng transaction để thực thi SQL
    try:
        with engine.begin() as conn: # Tự động commit nếu không lỗi, rollback nếu có lỗi
            for table_key, table_info in schemas.items():
                schema_name = table_info.get("db_schema", "raw")
                table_name = table_info.get("table_name", table_key)
                columns = table_info.get("columns", {})
                
                # BẢO MẬT: Bọc identifier bằng ngoặc kép chống SQL Injection và lỗi format (VD: khoảng trắng, gạch ngang)
                safe_schema = f'"{schema_name}"'
                safe_table = f'"{table_name}"'
                full_table_name = f"{safe_schema}.{safe_table}"
                
                logger.info(f"Đang xử lý DDL cho bảng {full_table_name}...")
                
                # B1: Đảm bảo Schema tồn tại
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {safe_schema};"))
                
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
                conn.execute(text(drop_sql))
                conn.execute(text(create_sql))
                
                logger.info(f"✅ Đã tạo bảng {full_table_name} thành công!")
                
    except Exception as e:
        logger.error(f"❌ Lỗi khi thực thi khởi tạo bảng: {e}")

def process_target_path(target_path: Path):
    """
    Hỗ trợ xử lý thông minh: 
    Nếu truyền vào 1 file -> Nạp file đó.
    Nếu truyền vào 1 thư mục -> Nạp tất cả các file .yaml bên trong.
    """
    if target_path.is_file() and target_path.suffix in ['.yaml', '.yml']:
        logger.info(f"\n--- Bắt đầu xử lý file: {target_path.name} ---")
        create_tables_from_yaml(target_path)
        
    elif target_path.is_dir():
        logger.info(f"\n--- Bắt đầu quét thư mục schema: {target_path} ---")
        yaml_files = list(target_path.glob("*.yaml")) + list(target_path.glob("*.yml"))
        
        if not yaml_files:
            logger.warning("Không tìm thấy file .yaml nào trong thư mục này.")
            return
            
        for y_file in yaml_files:
            create_tables_from_yaml(y_file)
            
    else:
        logger.error(f"❌ Đường dẫn không tồn tại: {target_path}")


if __name__ == "__main__":
    # Điểm thay đổi: Giờ đây biến này có thể trỏ thẳng tới THƯ MỤC chứa schema
    # (Hữu ích khi sau này bạn tách ra thành apple_schema.yaml, fitabase_schema.yaml, ...)
    schema_target_path = PROJECT_ROOT / "src" / "api" / "data_schema"
    
    # Chạy hàm xử lý động
    process_target_path(schema_target_path)