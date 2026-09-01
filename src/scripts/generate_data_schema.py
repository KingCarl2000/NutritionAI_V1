import pandas as pd
import yaml
import os
import glob

# 1. Dictionary cấu hình danh sách các dataset
# Dễ dàng thêm, sửa, xoá các dataset trong tương lai tại đây.
# Tính năng mới: 'data_path' bây giờ có thể là ĐƯỜNG DẪN THƯ MỤC hoặc ĐƯỜNG DẪN FILE.
DATASETS_CONFIG = {
    # Ví dụ 1: Quét toàn bộ thư mục (dành cho bộ dữ liệu Fitabase gồm nhiều file)
    "fitabase_tracker_data": {
        "data_path": r"D:\NutritionAI_V1\Data\raw\Fitabase Data 3.12.16-4.11.16",
        "schema_path": r"D:\NutritionAI_V1\src\api\data_schema\schema.yaml",
        "db_schema": "raw"
    },
    
    # Ví dụ 2 (Đóng comment): Nạp 1 file đơn lẻ (tương lai)
    # "apple_data_raw": {
    #     "data_path": r"D:\NutritionAI_V1\Data\raw\apple_health_export.csv",
    #     "schema_path": r"D:\NutritionAI_V1\src\api\data_schema\apple_schema.yaml",
    #     "db_schema": "staging"
    # }
}

def process_single_csv(file_path, table_name_override, config, existing_schema):
    """
    Hàm xử lý một file CSV duy nhất và cập nhật vào existing_schema.
    - file_path: Đường dẫn thực tế đến file CSV.
    - table_name_override: Tên bảng (lấy từ config gốc nếu là file đơn, lấy từ tên file nếu duyệt thư mục).
    """
    db_schema_name = config.get("db_schema", "raw")
    print(f"  -> Bắt đầu phân tích file: '{os.path.basename(file_path)}' thành bảng '{table_name_override}'")
    
    # Đọc tập dữ liệu
    try:
        df = pd.read_csv(file_path)
        print(f"     Đã tải thành công dataset với {df.shape[0]} dòng và {df.shape[1]} cột.")
    except Exception as e:
        print(f"     ❌ Lỗi khi đọc file {file_path}: {e} -> Bỏ qua.")
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
    # Gom nhóm theo schema_path để tối ưu việc đọc/ghi YAML
    # Nếu nhiều dataset cấu hình chung 1 file yaml, ta chỉ cần mở/ghi file đó 1 lần.
    yaml_groups = {}
    for ds_name, config in DATASETS_CONFIG.items():
        yaml_path = config["schema_path"]
        if yaml_path not in yaml_groups:
            yaml_groups[yaml_path] = {}
            # Đọc nội dung file yaml cũ (nếu có)
            if os.path.exists(yaml_path):
                with open(yaml_path, "r", encoding="utf-8") as file:
                    yaml_groups[yaml_path] = yaml.safe_load(file) or {}
            else:
                os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    # Duyệt qua từng cấu hình dataset
    for config_name, config in DATASETS_CONFIG.items():
        data_path = config["data_path"]
        schema_path = config["schema_path"]
        
        print(f"\n=== Xử lý cấu hình: '{config_name}' ===")
        
        if not os.path.exists(data_path):
            print(f"❌ Đường dẫn không tồn tại: {data_path}")
            continue

        # Lấy dữ liệu schema hiện tại của file yaml tương ứng
        current_yaml_data = yaml_groups[schema_path]

        # Kiểm tra xem đường dẫn là THƯ MỤC hay FILE
        if os.path.isdir(data_path):
            print(f"Phát hiện đây là THƯ MỤC. Đang quét các file .csv...")
            csv_files = glob.glob(os.path.join(data_path, "*.csv"))
            if not csv_files:
                print("   Không có file .csv nào trong thư mục này.")
            
            for file_path in csv_files:
                # Tên bảng lấy theo tên file csv
                table_name = os.path.splitext(os.path.basename(file_path))[0]
                current_yaml_data = process_single_csv(file_path, table_name, config, current_yaml_data)
                
        elif os.path.isfile(data_path) and data_path.lower().endswith('.csv'):
            print(f"Phát hiện đây là FILE csv đơn lẻ.")
            # Tên bảng lấy theo tên cấu hình (config_name)
            current_yaml_data = process_single_csv(data_path, config_name, config, current_yaml_data)
            
        else:
            print(f"❌ Đường dẫn không hợp lệ (không phải thư mục cũng không phải file csv).")

        # Cập nhật lại dữ liệu vào nhóm
        yaml_groups[schema_path] = current_yaml_data

    # Ghi lại tất cả các file YAML đã bị thay đổi
    print("\n--- HOÀN TẤT PHÂN TÍCH. ĐANG LƯU KẾT QUẢ ---")
    for yaml_path, yaml_data in yaml_groups.items():
        with open(yaml_path, "w", encoding="utf-8") as file:
            yaml.dump(yaml_data, file, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"✅ Đã lưu thay đổi vào: {yaml_path}")


if __name__ == "__main__":
    generate_schemas()