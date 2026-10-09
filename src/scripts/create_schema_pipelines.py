import os
import yaml
import json
from pathlib import Path

def generate_pipeline_schemas_from_config(pipelines_config: list, yaml_report_path: str = "docs/architecture/tables_created_report.yaml"):
    """
    Đọc file tables_created_report.yaml, tự động phân loại cột (numerical/categorical)
    và xuất ra các file JSON schema tương ứng cho từng pipeline.
    """
    yaml_path = Path(yaml_report_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file báo cáo schema tại: {yaml_path.absolute()}")

    print(f"Đang đọc cấu trúc chung từ file YAML: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    raw_tables = data.get('raw', {})

    # Duyệt qua từng cấu hình pipeline trong danh sách để xử lý độc lập
    for cfg in pipelines_config:
        source_dataset = cfg.get("source_dataset")
        dir_path = cfg.get("dir_path")
        index_col = cfg.get("index")
        drop_cols = cfg.get("drop_cols", [])
        table_name = cfg.get("table_name", source_dataset)

        if table_name not in raw_tables:
            print(f"[CẢNH BÁO] Không tìm thấy bảng '{table_name}' trong file YAML cho pipeline '{source_dataset}'. Bỏ qua.")
            continue

        columns_config = raw_tables[table_name].get('columns', {})

        # TỰ ĐỘNG PHÂN LOẠI CỘT TỪ YAML
        numerical_cols = []
        categorical_cols = []

        for col_name, props in columns_config.items():
            # Bỏ qua cột khóa chính hoặc các cột nằm trong danh sách drop tùy ý nếu cần, 
            # hoặc giữ nguyên phân loại theo yaml_type của file YAML:
            yaml_type = props.get('yaml_type', 'categorical')
            
            if yaml_type in ['integer', 'float']:
                numerical_cols.append(col_name)
            elif yaml_type == 'categorical':
                categorical_cols.append(col_name)

        # Xây dựng cấu trúc dữ liệu JSON hoàn chỉnh cho pipeline tương ứng
        schema_data = {
            "source_dataset": source_dataset,
            "index": index_col,
            "drop_columns": drop_cols,
            "columns": columns_config,
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols
        }

        # Đảm bảo thư mục đầu ra tồn tại
        out_path = Path(dir_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Ghi nội dung ra file JSON đích
        out_path.write_text(json.dumps(schema_data, indent=4, ensure_ascii=False), encoding='utf-8')
        print(f"[THÀNH CÔNG] Đã tạo schema cho pipeline '{source_dataset}' tại -> {out_path.absolute()}")
        print(f"   -> Numerical: {numerical_cols}")
        print(f"   -> Categorical: {categorical_cols}\n")

if __name__ == "__main__":
    # NƠI CẤU HÌNH DANH SÁCH PIPELINE (Chỉ cần khai báo dataset, đường dẫn lưu và cột index/drop)
    PIPELINES_REGISTRY = [
        {
            "source_dataset": "calories",
            "dir_path": r"D:\NutritionAI_V1\config\pipelines\ml_calories\calories_data_schema.json",
            "index": "User_ID",
            "drop_cols": ["User_ID"]
        },
        # =========================================================================
        # MỞ RỘNG THÊM PIPELINE / DATASET MỚI TRONG TƯƠNG LAI Ở ĐÂY:
        # {
        #     "source_dataset": "ten_dataset_moi",
        #     "table_name": "ten_bang_moi",
        #     "dir_path": r"D:\NutritionAI_V1\config\pipelines\ml_moi\schema_moi.json",
        #     "index": "Cột_ID",
        #     "drop_cols": ["Cột_ID"]
        # }
        # =========================================================================
    ]

    print("=== BẮT ĐẦU QUÁ TRÌNH TỰ ĐỘNG TẠO SCHEMA CHO CÁC PIPELINE ===")
    generate_pipeline_schemas_from_config(PIPELINES_REGISTRY)
    print("=== HOÀN TẤT TẤT CẢ PIPELINES ===")