import pandas as pd
import yaml
import os
import glob
from pathlib import Path
from src.nutrition_core.logging.logger import logger

# 1. Tự động lấy đường dẫn gốc của dự án (NutritionAI_V1)
# Đường dẫn hiện tại: src/scripts/generate_data_schema.py -> lùi 2 cấp là về gốc
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 2. Dictionary cấu hình danh sách các dataset sử dụng đường dẫn ĐỘNG
DATASETS_CONFIG = {
    "fitabase_tracker_data": {
        "data_path": str(PROJECT_ROOT / "Data" / "raw" / "Fitabase Data 3.12.16-4.11.16"),
        "schema_path": str(PROJECT_ROOT / "src" / "api" / "data_schema" / "schema.yaml"),
        "db_schema": "raw"
    }
}

def process_single_csv(file_path, table_name_override, config, existing_schema):
    """Hàm xử lý một file CSV duy nhất và cập nhật vào existing_schema."""
    db_schema_name = config.get("db_schema", "raw")
    logger.info(f"Bắt đầu phân tích file: '{os.path.basename(file_path)}' thành bảng '{table_name_override}'")

    # Đọc tập dữ liệu
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Đã tải thành công dataset với {df.shape[0]} dòng và {df.shape[1]} cột.")
    except Exception as e:
        logger.error(f"Lỗi khi đọc file {file_path}: {e} -> Bỏ qua.")
        return existing_schema

    # Khởi tạo cấu trúc Schema
    schema_data = {
        "db_schema": db_schema_name,
        "table_name": table_name_override,
        "columns": {},
        "numerical_columns": [],
        "categorical_columns": [],
        "datetime_columns": [],
        "boolean_columns": []
    }

    # Trích xuất columns và phân loại
    if not df.empty:
        for col_name, dtype in df.dtypes.items():
            if pd.api.types.is_integer_dtype(dtype):
                col_type = "integer"
                schema_data["numerical_columns"].append(col_name)
            elif pd.api.types.is_float_dtype(dtype):
                col_type = "float"
                schema_data["numerical_columns"].append(col_name)
            elif pd.api.types.is_bool_dtype(dtype):
                col_type = "boolean"
                schema_data["boolean_columns"].append(col_name)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                col_type = "datetime"
                schema_data["datetime_columns"].append(col_name)
            else:
                col_type = "categorical"
                schema_data["categorical_columns"].append(col_name)

            schema_data["columns"][col_name] = {
                "type": col_type,
                "nullable": bool(df[col_name].isnull().any().item()),
                "description": ""
            }

    # Loại bỏ các list rỗng
    schema_data = {k: v for k, v in schema_data.items() if v != []}

    # Cập nhật schema
    existing_schema[table_name_override] = schema_data
    return existing_schema


def generate_schemas():
    yaml_groups = {}
    
    for ds_name, config in DATASETS_CONFIG.items():
        yaml_path = config["schema_path"]
        if yaml_path not in yaml_groups:
            yaml_groups[yaml_path] = {}

    # Đọc nội dung file yaml cũ (nếu có)
    for yaml_path in yaml_groups.keys():
        if os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as file:
                yaml_groups[yaml_path] = yaml.safe_load(file) or {}
        else:
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    # Duyệt qua từng cấu hình dataset
    for config_name, config in DATASETS_CONFIG.items():
        data_path = config["data_path"]
        schema_path = config["schema_path"]
        
        logger.info(f"=== Xử lý cấu hình: '{config_name}' ===")

        if not os.path.exists(data_path):
            logger.error(f"Đường dẫn không tồn tại: {data_path}")
            continue

        current_yaml_data = yaml_groups[schema_path]

        if os.path.isdir(data_path):
            logger.info("Phát hiện đây là THƯ MỤC. Đang quét các file .csv...")
            folder_name = os.path.basename(os.path.normpath(data_path))
            
            if folder_name not in current_yaml_data:
                current_yaml_data[folder_name] = {}
                
            csv_files = glob.glob(os.path.join(data_path, "*.csv"))
            if not csv_files:
                logger.warning("Không có file .csv nào trong thư mục này.")
                
            for file_path in csv_files:
                table_name = os.path.splitext(os.path.basename(file_path))[0]
                current_yaml_data[folder_name] = process_single_csv(
                    file_path, table_name, config, current_yaml_data[folder_name]
                )
                
        elif os.path.isfile(data_path) and data_path.lower().endswith('.csv'):
            logger.info("Phát hiện đây là FILE csv đơn lẻ.")
            if config_name not in current_yaml_data:
                current_yaml_data[config_name] = {}
            current_yaml_data[config_name] = process_single_csv(
                data_path, config_name, config, current_yaml_data[config_name]
            )
        else:
            logger.error("Đường dẫn không hợp lệ (không phải thư mục cũng không phải file csv).")

        yaml_groups[schema_path] = current_yaml_data

    # Ghi lại tất cả các file YAML đã bị thay đổi
    logger.info("--- HOÀN TẤT PHÂN TÍCH. ĐANG LƯU KẾT QUẢ ---")
    for yaml_path, yaml_data in yaml_groups.items():
        with open(yaml_path, "w", encoding="utf-8") as file:
            yaml.dump(yaml_data, file, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info(f"Đã lưu thay đổi vào: {yaml_path}")


if __name__ == "__main__":
    generate_schemas()