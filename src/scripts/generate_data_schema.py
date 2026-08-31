import pandas as pd
import yaml
import os

# 1. Dictionary cấu hình danh sách các dataset
# Dễ dàng thêm, sửa, xoá các dataset trong tương lai tại đây
DATASETS_CONFIG = {
    "fitness_tracker_dataset": {
        "data_path": r"D:\NutritionAI_V1\Data\raw\fitness_tracker_dataset.csv",
        "schema_path": r"D:\NutritionAI_V1\src\api\data_schema\schema.yaml",
        "db_schema": "raw"
    },
    # Mẫu cho dataset tương lai (có thể dùng chung file schema.yaml hoặc tách file riêng):
    # "apple_data_raw": {
    #     "data_path": r"D:\NutritionAI_V1\Data\raw\apple_health_export.csv",
    #     "schema_path": r"D:\NutritionAI_V1\src\api\data_schema\schema.yaml",
    #     "db_schema": "raw"
    # }
}

def generate_schemas():
    for table_name, config in DATASETS_CONFIG.items():
        data_path = config["data_path"]
        schema_path = config["schema_path"]
        db_schema_name = config.get("db_schema", "raw")
        
        print(f"\n=== Bắt đầu phân tích Schema cho bảng: '{table_name}' ===")
        
        # 2. Đọc tập dữ liệu
        try:
            df = pd.read_csv(data_path)
            print(f"Đã tải thành công dataset với {df.shape[0]} dòng và {df.shape[1]} cột.")
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file dữ liệu tại: {data_path} -> Bỏ qua bảng này.")
            continue # Bỏ qua và chạy tiếp dataset tiếp theo

        # 3. Khởi tạo cấu trúc Schema cho bảng hiện tại
        schema_data = {
            "db_schema": db_schema_name,
            "table_name": table_name,
            "columns": {},
            "numerical_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "boolean_columns": []
        }

        # 4. Trích xuất columns và phân loại
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
                    "nullable": bool(df[col_name].isnull().any().item()), # Ép sang bool chuẩn an toàn
                    "description": "" 
                }

        # 5. Loại bỏ các list rỗng (nhưng giữ nguyên chuỗi string)
        schema_data = {k: v for k, v in schema_data.items() if v != []}

        # 6. Đọc nội dung schema cũ (nếu có) để không bị ghi đè mất cấu hình các bảng khác
        existing_schema = {}
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as file:
                existing_schema = yaml.safe_load(file) or {}
        else:
            os.makedirs(os.path.dirname(schema_path), exist_ok=True)

        # Cập nhật schema của bảng hiện tại vào file tổng (dùng table_name làm Key gốc)
        existing_schema[table_name] = schema_data

        # 7. Ghi ra file YAML
        with open(schema_path, "w", encoding="utf-8") as file:
            yaml.dump(existing_schema, file, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"✅ Đã lưu cấu hình '{table_name}' vào: {schema_path}")

if __name__ == "__main__":
    generate_schemas()