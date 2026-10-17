from pathlib import Path
from typing import Type, Dict, Any
import yaml
from sqlalchemy import Column, Integer, String, Float, Boolean, Date, DateTime
from src.nutrition_core.database.session import Base
from src.nutrition_core.logging.logger import logger

# Xác định thư mục cấu hình config/datasets/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "config" / "datasets"

# Ánh xạ kiểu dữ liệu từ YAML sang SQLAlchemy Types
TYPE_MAPPING = {
    "integer": Integer,
    "string": String,
    "float": Float,
    "boolean": Boolean,
    "date": Date,
    "datetime": DateTime,
}

def build_model_from_yaml(yaml_filename: str) -> Type[Base]:
    """
    Đọc tệp YAML và tự động đăng ký Model động vào Base.metadata.
    Đã hỗ trợ xử lý nested dictionary (có root key) và các thuộc tính cột.
    """
    file_path = CONFIG_DIR / yaml_filename if not Path(yaml_filename).is_absolute() else Path(yaml_filename)
    
    if not file_path.suffix:
        file_path = file_path.with_suffix(".yaml")
    if not file_path.exists():
        logger.error(f"Không tìm thấy file cấu hình dataset: {file_path}")
        raise FileNotFoundError(f"Missing configuration file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # 1. FIX LỖI ROOT KEY: Nếu YAML có bọc trong root key (vd: 'calories': {...})
    dataset_key = list(raw_config.keys())[0]
    config = raw_config[dataset_key] if isinstance(raw_config.get(dataset_key), dict) else raw_config

    table_name = config.get("table_name", dataset_key)
    schema_name = config.get("db_schema", "raw")
    columns_info = config.get("columns", {})

    class_name = f"{table_name.capitalize()}Raw"
    attrs: Dict[str, Any] = {
        "__tablename__": table_name,
        "__table_args__": {"schema": schema_name},
        "__module__": __name__,
    }

    has_pk = False
    
    # 2. FIX LỖI COLUMN DICT: Xử lý cột chứa các thuộc tính {type, nullable...}
    for col_name, col_attrs in columns_info.items():
        # Xử lý tương thích ngược: nếu YAML khai báo kiểu string đơn thuần thay vì dict
        if isinstance(col_attrs, dict):
            col_type_str = col_attrs.get("type", "string").lower()
            is_nullable = col_attrs.get("nullable", True)
        else:
            col_type_str = str(col_attrs).lower()
            is_nullable = True
            
        # Ánh xạ kiểu categorical sang String cho SQLAlchemy
        if col_type_str == "categorical":
            col_type_str = "string"

        sa_type = TYPE_MAPPING.get(col_type_str, String)

        # Nhận diện khóa chính
        if col_name.lower() in ("user_id", "id") and not has_pk:
            attrs[col_name] = Column(sa_type, primary_key=True, index=True)
            has_pk = True
        else:
            attrs[col_name] = Column(sa_type, nullable=is_nullable)

    if not has_pk:
        attrs["id"] = Column(Integer, primary_key=True, autoincrement=True)

    model_class = type(class_name, (Base,), attrs)
    logger.info(f"Đã đăng ký thành công Model động '{class_name}' cho schema '{schema_name}.{table_name}'")
    return model_class

# ==========================================
# MODULE 1: NUTRITION & DIET CORE MODELS
# ==========================================
# Mở rộng trong tương lai cho các bảng domain dinh dưỡng cố định (Ingredients, Recipes, DietLogs)


# ==========================================
# MODULE 2: PIPELINE DATASETS (Chỉ định thủ công)
# ==========================================
# Giai đoạn hiện tại: Chỉ nạp và quản lý schema cho calories.yaml
CaloriesRaw = build_model_from_yaml("calories.yaml")