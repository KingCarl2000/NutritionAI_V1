import yaml
import json
import re
from pathlib import Path
from src.nutrition_core.logging.logger import logger
from src.postgres.core.connection import get_connection

def normalize_identifier(name: str) -> str:
    """Biến tên folder (VD: 'Fitabase Data 3.12') thành prefix an toàn cho PostgreSQL."""
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', name).lower()
    return re.sub(r'_+', '_', clean_name).strip('_')

def get_pg_data_type(yaml_type: str | None) -> str:
    """Maps generic YAML data types to PostgreSQL data types."""
    if not yaml_type:
        return "TEXT"
    type_mapping = {
        "string": "VARCHAR(255)",
        # 🔄 Đổi INTEGER thành BIGINT để chứa các ID dạng timestamp hoặc số lớn
        "int": "BIGINT",
        "integer": "BIGINT",
        "float": "DOUBLE PRECISION",
        "double": "DOUBLE PRECISION",
        "boolean": "BOOLEAN",
        "bool": "BOOLEAN",
        "date": "DATE",
        "datetime": "TIMESTAMP",
        "text": "TEXT"
    }
    return type_mapping.get(yaml_type.lower(), "TEXT")

def create_tables_from_yaml(yaml_path: Path, report_output_path: Path = None):
    """Đọc file schema.yaml, tạo các bảng trong PostgreSQL và xuất báo cáo schema đã tạo."""
    if not yaml_path.exists():
        logger.error(f"❌ Không tìm thấy file schema tại: {yaml_path}")
        return

    # 1. Đọc nội dung file schema.yaml gốc
    with open(yaml_path, 'r', encoding='utf-8') as f:
        schemas = yaml.safe_load(f)

    if not schemas:
        logger.warning(f"File schema trống hoặc không hợp lệ: {yaml_path.name}")
        return

    # Từ điển lưu lại thông tin các bảng thực tế đã được khởi tạo thành công để xuất file report
    created_tables_report = {}

    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                for dataset_name, tables_dict in schemas.items():
                    logger.info(f"📂 Đang xử lý nhóm dữ liệu: {dataset_name}")
                    
                    dataset_prefix = normalize_identifier(dataset_name)

                    if "columns" in tables_dict:
                        tables_dict = {dataset_name: tables_dict}
                        dataset_prefix = "" 

                    for table_key, table_info in tables_dict.items():
                        schema_name = table_info.get("db_schema", "raw")
                        base_table = table_info.get("table_name", table_key)
                        columns = table_info.get("columns", {})
                        
                        # Chuẩn hóa tên bảng thực tế trong Postgres
                        if dataset_prefix and not base_table.lower().startswith(dataset_prefix):
                            final_table_name = f"{dataset_prefix}_{base_table}"
                        else:
                            final_table_name = base_table
                            
                        safe_schema = f'"{schema_name}"'
                        safe_table = f'"{final_table_name}"'
                        full_table_name = f"{safe_schema}.{safe_table}"
                        
                        logger.info(f"Đang xử lý DDL cho bảng {full_table_name}...")

                        # B1: Đảm bảo Schema tồn tại
                        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {safe_schema};")
                        
                        # B2: Xây dựng danh sách các cột an toàn
                        col_definitions = []
                        column_details_report = {}
                        
                        for col_name, col_attrs in columns.items():
                            yaml_type = col_attrs.get("type", "TEXT")
                            pg_type = get_pg_data_type(yaml_type)
                            safe_col_name = f'"{col_name}"'
                            col_definitions.append(f"{safe_col_name} {pg_type}")
                            
                            # Lưu thông tin chi tiết cột vào báo cáo
                            column_details_report[col_name] = {
                                "yaml_type": yaml_type,
                                "postgresql_type": pg_type,
                                "nullable": col_attrs.get("nullable", True)
                            }
                            
                        # B3: Xây dựng câu lệnh SQL DDL hoàn chỉnh
                        columns_ddl = ",\n    ".join(col_definitions)
                        
                        drop_sql = f"DROP TABLE IF EXISTS {full_table_name} CASCADE;"
                        create_sql = f"""CREATE TABLE {full_table_name} (\n    {columns_ddl}\n);"""
                        
                        # B4: Thực thi SQL
                        cursor.execute(drop_sql)
                        cursor.execute(create_sql)
                        
                        # Lưu vào báo cáo tổng hợp
                        if schema_name not in created_tables_report:
                            created_tables_report[schema_name] = {}
                            
                        created_tables_report[schema_name][final_table_name] = {
                            "source_dataset": dataset_name,
                            "columns": column_details_report
                        }
                        
                        logger.info(f"✅ Đã tạo bảng {full_table_name} thành công!")
                        
            conn.commit()
            logger.info("🎉 Toàn bộ các bảng đã được khởi tạo và commit thành công!")

            # 5. Xuất file báo cáo thông tin bảng đã tạo ra file YAML
            if report_output_path:
                report_output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(report_output_path, 'w', encoding='utf-8') as report_f:
                    yaml.dump(created_tables_report, report_f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                logger.info(f"📝 Đã xuất báo cáo danh sách bảng ra file: {report_output_path}")
            
    except Exception as e:
        logger.error(f"❌ Lỗi khi thực thi khởi tạo bảng: {e}")
        raise e

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SCHEMA_PATH = PROJECT_ROOT / "src" / "api" / "data_schema" / "schema.yaml"
    
    # Định nghĩa đường dẫn xuất file thông tin bảng đã tạo
    REPORT_PATH = PROJECT_ROOT / "docs" / "architecture" / "tables_created_report.yaml"
    
    logger.info("="*50)
    logger.info("🚀 BẮT ĐẦU QUÁ TRÌNH TẠO BẢNG POSTGRESQL & XUẤT BÁO CÁO")
    logger.info(f"Đọc cấu hình từ: {SCHEMA_PATH}")
    logger.info("="*50)
    
    create_tables_from_yaml(SCHEMA_PATH, report_output_path=REPORT_PATH)