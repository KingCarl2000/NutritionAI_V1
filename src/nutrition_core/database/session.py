import os
import yaml
from pathlib import Path
from sqlalchemy import create_engine, text, Column, Integer, String, Float, Boolean, Date, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

# Import engine từ dự án của bạn (sửa đường dẫn nếu cần)
try:
    from src.postgres.core.connection import get_engine
except ImportError:
    # Fallback nếu chưa cấu hình engine ở connection.py
    engine = None 

# Import Custom Logger của bạn
from src.nutrition_core.logging.logger import logger

# ==========================================
# 1. CẤU HÌNH CORE SESSION & BASE METADATA
# ==========================================
# autocommit=False, autoflush=False để kiểm soát giao dịch an toàn
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())

# Base class để theo dõi toàn bộ Table sẽ được tạo
Base = declarative_base()

# Trỏ đến thư mục chứa config datasets (config/datasets/...)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "config" / "datasets"


# ==========================================
# 2. DYNAMIC TABLE BUILDER TỪ YAML
# ==========================================
def load_yaml_config(file_name: str) -> dict:
    """Hàm đọc nội dung file cấu hình YAML."""
    if not file_name.endswith('.yaml'):
        file_name += '.yaml'
        
    config_path = CONFIG_DIR / file_name
    if not config_path.exists():
        logger.error(f"Không tìm thấy file config pipeline: {config_path}")
        raise FileNotFoundError(f"Missing configuration file: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def map_sqlalchemy_type(type_str: str):
    """Chuyển đổi kiểu dữ liệu quy định trong YAML sang chuẩn SQLAlchemy."""
    mapping = {
        "integer": Integer,
        "string": String,
        "float": Float,
        "boolean": Boolean,
        "date": Date,
        "datetime": DateTime
    }
    return mapping.get(type_str.lower(), String)

def register_pipeline_model(dataset_name: str):
    """
    Đọc config YAML và tự động đăng ký Model động vào Base.metadata.
    (Hỗ trợ đắc lực cho Alembic tự động quét schema và tạo bảng DDL).
    """
    config = load_yaml_config(dataset_name)
    
    table_name = config.get("table_name", dataset_name)
    # Theo sơ đồ Ingestion, dữ liệu thô thường vào schema 'raw' hoặc 'staging'
    schema_name = config.get("schema", "raw") 
    columns_info = config.get("columns", {})

    # Các thuộc tính cốt lõi của Class Model SQLAlchemy
    attrs = {
        '__tablename__': table_name,
        '__table_args__': {'schema': schema_name},
        '__module__': __name__
    }

    # Sinh các cột tự động dựa trên YAML
    for col_name, col_props in columns_info.items():
        col_type = map_sqlalchemy_type(col_props.get("type", "string"))
        is_pk = col_props.get("primary_key", False)
        is_nullable = col_props.get("nullable", True)
        
        attrs[col_name] = Column(col_type, primary_key=is_pk, nullable=is_nullable)

    # Khởi tạo Class Model động lúc Runtime (VD: 'calories' -> CaloriesModel)
    class_name = "".join(word.capitalize() for word in table_name.split('_')) + "Model"
    DynamicModel = type(class_name, (Base,), attrs)
    
    logger.info(f"Đã đăng ký model động: {class_name} (Lưu tại: {schema_name}.{table_name})")
    
    return DynamicModel, config


# ==========================================
# 3. TRÌNH QUẢN LÝ PHIÊN LÀM VIỆC (DEPENDENCIES)
# ==========================================
def get_db():
    """Cấp phát session tiêu chuẩn, dùng chung cho API hoặc tác vụ CRUD cơ bản."""
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Lỗi truy vấn Database (SQLAlchemyError): {str(e)}")
        db.rollback()
        raise
    except Exception as e:
        logger.error(f"Lỗi hệ thống trong phiên làm việc DB: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()

def get_pipeline_session(dataset_name: str):
    """
    Session Generator dành riêng cho Data Ingestion Pipeline.
    Tự động đăng ký Table Model từ YAML, chuyển context của session vào đúng Schema 
    và trả về Session cùng với cấu hình của Pipeline đó.
    """
    # Bước 1: Chuẩn bị bảng động
    _, config = register_pipeline_model(dataset_name)
    target_schema = config.get("schema", "raw")
    
    db = SessionLocal()
    try:
        # Bước 2: Chỉ định search_path để mọi câu lệnh SQL trong session này 
        # đều ưu tiên target_schema (raw/staging)
        db.execute(text(f"SET search_path TO {target_schema}"))
        
        # Trả về db để thao tác và config để các Script MLOps/ETL biết cần xử lý gì
        yield db, config
    except SQLAlchemyError as e:
        logger.error(f"Lỗi Database trong Pipeline '{dataset_name}': {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()