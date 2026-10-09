import os
import yaml
from sqlalchemy import Column, Integer, String, Float, BigInteger, Boolean
from src.nutrition_core.database.session import Base

# 1. Từ điển ánh xạ từ postgresql_type (trong YAML) sang SQLAlchemy Type
PG_TO_SA_TYPE = {
    'BIGINT': BigInteger,
    'DOUBLE PRECISION': Float,
    'TEXT': String,
    'INTEGER': Integer,
    'BOOLEAN': Boolean,
    # Thêm các kiểu khác trong tương lai nếu YAML của bạn có (vd: JSONB, TIMESTAMP...)
}

# 2. Đường dẫn tới file YAML
# Giả sử models.py nằm ở src/nutrition_core/database/
# Ta lùi về thư mục gốc để trỏ tới docs/architecture/...
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
YAML_PATH = os.path.join(BASE_DIR, "docs", "architecture", "tables_created_report.yaml")

# 3. Hàm tự động sinh Models từ YAML
def generate_models_from_yaml():
    if not os.path.exists(YAML_PATH):
        print(f"Cảnh báo: Không tìm thấy file YAML tại {YAML_PATH}")
        return

    with open(YAML_PATH, 'r', encoding='utf-8') as file:
        yaml_data = yaml.safe_load(file)

    # Lặp qua từng schema (vd: 'raw')
    for schema_name, tables in yaml_data.items():
        if not isinstance(tables, dict):
            continue
            
        # Lặp qua từng bảng trong schema (vd: 'calories')
        for table_name, table_info in tables.items():
            
            # Chuẩn bị thuộc tính cho Class
            class_attrs = {
                '__tablename__': table_name,
                '__table_args__': {'schema': schema_name},
                # Mặc định thêm cột id làm Primary Key cho mọi bảng
                'id': Column(Integer, primary_key=True, autoincrement=True)
            }

            # Lặp qua từng cột để tạo SQLAlchemy Column
            columns = table_info.get('columns', {})
            for col_name, col_props in columns.items():
                # Lấy kiểu dữ liệu và map sang SQLAlchemy
                pg_type = col_props.get('postgresql_type', 'TEXT').upper()
                sa_type = PG_TO_SA_TYPE.get(pg_type, String) # Mặc định là String nếu không tìm thấy
                
                is_nullable = col_props.get('nullable', True)
                
                # Logic đánh Index: Ví dụ mặc định đánh index cho cột có chữ 'ID'
                is_index = True if 'ID' in col_name.upper() else False

                # Đưa cột vào thuộc tính của class
                class_attrs[col_name] = Column(sa_type, nullable=is_nullable, index=is_index)

            # Khởi tạo Class động (Tên class viết hoa chữ cái đầu: calories -> Calories)
            class_name = ''.join(word.capitalize() for word in table_name.split('_'))
            
            # Hàm type() của Python dùng để tạo Class động và gán vào Base của SQLAlchemy
            # Cú pháp: type(TênClass, (Các_Class_Kế_Thừa,), Các_Thuộc_Tính)
            type(class_name, (Base,), class_attrs)

# 4. Thực thi hàm khi file models.py được import
generate_models_from_yaml()