import pandas as pd
import yaml
import os

# 1. Khai báo đường dẫn
# Sửa lại đường dẫn này trỏ tới file dữ liệu (.csv)
data_path = r"D:\NutritionAI_V1\Data\raw\health_data.csv"
schema_path = r"D:\NutritionAI_V1\src\api\data_schema\schema.yaml"

# 2. Đọc tập dữ liệu
try:
    df = pd.read_csv(data_path)
    print(f"Đã tải thành công dataset với {df.shape[0]} dòng và {df.shape[1]} cột.")
except FileNotFoundError:
    print(f"❌ Không tìm thấy file dữ liệu tại: {data_path}")
    df = pd.DataFrame() 

# 3. Khởi tạo cấu trúc Schema
schema_data = {
    "columns": {},
    "numerical_columns": [],
    "categorical_columns": [],
    "datetime_columns": [],
    "boolean_columns": []
}

# 4. Trích xuất columns và phân loại vào các nhóm
if not df.empty:
    for col_name, dtype in df.dtypes.items():
        # -- Phân loại kiểu dữ liệu --
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
            col_type = "categorical" # Xử lý Object/String
            schema_data["categorical_columns"].append(col_name)

        # -- Ghi chi tiết từng cột --
        schema_data["columns"][col_name] = {
            "type": col_type,
            "nullable": df[col_name].isnull().any().item(), # .item() để chuyển từ numpy bool sang python bool chuẩn
            "description": "" 
        }

# 5. Loại bỏ các list rỗng (nếu dataset không có kiểu dữ liệu đó)
# Ví dụ: Nếu không có cột boolean, nó sẽ không in ra "boolean_columns: []"
schema_data = {k: v for k, v in schema_data.items() if v}

# 6. Đọc nội dung schema cũ (nếu có) để giữ lại các config tuỳ chỉnh khác (như target_column)
if os.path.exists(schema_path):
    with open(schema_path, "r", encoding="utf-8") as file:
        existing_schema = yaml.safe_load(file) or {}
        # Hợp nhất: cập nhật các key mới tạo vào schema cũ
        existing_schema.update(schema_data)
        schema_data = existing_schema
else:
    os.makedirs(os.path.dirname(schema_path), exist_ok=True)

# 7. Ghi ra file YAML
with open(schema_path, "w", encoding="utf-8") as file:
    yaml.dump(schema_data, file, default_flow_style=False, allow_unicode=True, sort_keys=False)

print(f"✅ Đã tạo schema và phân nhóm thành công. Lưu tại: {schema_path}")