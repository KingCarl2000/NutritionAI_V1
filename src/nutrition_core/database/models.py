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
    Đọc tệp YAML được chỉ định thủ công và tự động đăng ký Model động vào Base.metadata.
    
    Args:
        yaml_filename (str): Tên file hoặc đường dẫn tương đối trong config/datasets/ (ví dụ: 'calories.yaml')
    Returns:
        Type[Base]: Đối tượng SQLAlchemy Model Class được tạo động.
    """
    file_path = CONFIG_DIR / yaml_filename if not Path(yaml_filename).is_absolute() else Path(yaml_filename)
    if not file_path.suffix:
        file_path = file_path.with_suffix(".yaml")

    if not file_path.exists():
        logger.error(f"Không tìm thấy file cấu hình dataset: {file_path}")
        raise FileNotFoundError(f"Missing configuration file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    table_name = config.get("table_name", file_path.stem)
    schema_name = config.get("db_schema", config.get("schema", "raw"))
    columns_info = config.get("columns", {})

    # Đặt tên Class Python (ví dụ: calories -> CaloriesRaw)
    class_name = f"{table_name.capitalize()}Raw"

    attrs: Dict[str, Any] = {
        "__tablename__": table_name,
        "__table_args__": {"schema": schema_name},
        "__module__": __name__,
    }

    # Sinh các thuộc tính Column dựa trên cấu hình YAML
    has_pk = False
    for col_name, col_type_str in columns_info.items():
        sa_type = TYPE_MAPPING.get(str(col_type_str).lower(), String)
        
        # Nhận diện khóa chính (Primary Key) cho User_ID hoặc ID
        if col_name.lower() in ("user_id", "id") and not has_pk:
            attrs[col_name] = Column(sa_type, primary_key=True, index=True)
            has_pk = True
        else:
            attrs[col_name] = Column(sa_type, nullable=True)

    # Fallback khóa chính nếu file YAML không có cột ID rõ ràng
    if not has_pk:
        attrs["id"] = Column(Integer, primary_key=True, autoincrement=True)

    # Khởi tạo class động kế thừa từ Base (Tự động đưa vào Base.metadata cho Alembic)
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