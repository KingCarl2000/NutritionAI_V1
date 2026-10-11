import os
import json
import glob
from sqlalchemy import Column, Integer, String, Float, BigInteger, Boolean
from src.nutrition_core.database.session import Base

# 1. Từ điển ánh xạ từ postgresql_type sang SQLAlchemy Type
PG_TO_SA_TYPE = {
    'BIGINT': BigInteger,
    'DOUBLE PRECISION': Float,
    'TEXT': String,
    'INTEGER': Integer,
    'BOOLEAN': Boolean,
    # Thêm các kiểu khác nếu cần
}

# 2. Đường dẫn tới thư mục gốc và thư mục chứa pipelines config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
PIPELINES_CONFIG_DIR = os.path.join(BASE_DIR, "config", "pipelines")

def generate_models_from_json():
    if not os.path.exists(PIPELINES_CONFIG_DIR):
        print(f"Cảnh báo: Không tìm thấy thư mục {PIPELINES_CONFIG_DIR}")
        return

    # Quét tất cả các file .json nằm sâu trong các thư mục con của config/pipelines/
    json_files = glob.glob(os.path.join(PIPELINES_CONFIG_DIR, "**", "*.json"), recursive=True)

    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Lỗi: Không thể đọc định dạng JSON từ file {json_path}")
            continue

        # Lấy tên bảng từ "source_dataset" (Nếu file không có key này -> không phải file schema -> bỏ qua)
        table_name = config_data.get("source_dataset")
        if not table_name:
            continue
            
        # Lấy cột sẽ làm khóa chính từ "index"
        pk_column_name = config_data.get("index")
        
        # Mặc định schema là public, nếu JSON của bạn có config "schema": "raw" thì thay thế
        schema_name = config_data.get("schema", "public")

        class_attrs = {
            '__tablename__': table_name,
            '__table_args__': {'schema': "nutrition_ai"}
        }

        # Giả định cấu trúc mô tả cột nằm trong key 'columns'
        columns = config_data.get("columns", {})
        for col_name, col_props in columns.items():
            pg_type = col_props.get('postgresql_type', 'TEXT').upper()
            sa_type = PG_TO_SA_TYPE.get(pg_type, String)
            is_nullable = col_props.get('nullable', True)
            
            # Nếu tên cột trùng với khai báo trong tham số "index", ta biến nó thành Primary Key
            is_pk = (col_name == pk_column_name)

            class_attrs[col_name] = Column(sa_type, primary_key=is_pk, nullable=is_nullable)

        # Khởi tạo Class động (VD: 'calories' -> 'Calories')
        class_name = ''.join(word.capitalize() for word in table_name.split('_'))
        
        # Ngăn ngừa việc tạo trùng class nếu nhiều JSON cùng trỏ về 1 bảng
        if class_name not in Base.registry._class_registry:
            type(class_name, (Base,), class_attrs)

# 3. Thực thi hàm khi file models.py được import
generate_models_from_json()