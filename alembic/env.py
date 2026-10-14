import logging
import sys
import os
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# 1. Thêm root path của project vào sys.path để import từ 'src'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# 2. Import Custom Logger của bạn
try:
    from src.nutrition_core.logging.logger import logger as custom_logger
except ImportError:
    custom_logger = logging.getLogger("alembic.env")

# 3. KÍCH HOẠT DYNAMIC MODELS TRƯỚC KHI LẤY METADATA
# Import Base và các thành phần từ session.py (bản cập nhật mới)
try:
    from src.nutrition_core.database.session import Base, register_pipeline_model, CONFIG_DIR
    
    # Quét toàn bộ file .yaml trong thư mục config/datasets/ và đăng ký Model
    if CONFIG_DIR.exists():
        for yaml_file in CONFIG_DIR.glob("*.yaml"):
            dataset_name = yaml_file.stem
            try:
                # Gọi hàm này sẽ tự động parse YAML và nhúng Model vào Base.metadata
                register_pipeline_model(dataset_name)
                custom_logger.info(f"[Alembic] Đã load cấu hình bảng từ {yaml_file.name}")
            except Exception as e:
                custom_logger.error(f"[Alembic] Lỗi load {yaml_file.name}: {e}")
                
    target_metadata = Base.metadata
except ImportError as e:
    target_metadata = None
    custom_logger.error(f"[Alembic] Không thể import từ session.py. Lỗi: {e}")

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 4. Tự động lấy URL từ src.postgres.core.connection
try:
    from src.postgres.core.connection import get_engine
    engine = get_engine()
    # Ghi đè URL trong alembic.ini bằng URL thực tế từ code hệ thống của bạn
    config.set_main_option("sqlalchemy.url", str(engine.url))
except Exception as e:
    custom_logger.warning(f"[Alembic] Cảnh báo: Không thể nạp database URL từ code. Sẽ dùng mặc định trong alembic.ini. Lỗi: {e}")

# 5. Cấu hình quản lý Multi-Schema (Dựa trên sơ đồ Data Ingestion)
TARGET_SCHEMAS = ["public", "raw", "staging"]

def include_name(name, type_, parent_names):
    """Giúp Alembic theo dõi các thay đổi trên nhiều schemas cụ thể."""
    if type_ == "schema":
        return name in TARGET_SCHEMAS
    else:
        return True

def run_migrations_offline() -> None:
    """Chạy migrations ở 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True, # Cho phép quét đa schema
        include_name=include_name,
        version_table_schema="public" 
    )

    with context.begin_transaction():
        context.run_migrations()
        custom_logger.info("Hoàn tất chạy migration (Offline mode)")

def run_migrations_online() -> None:
    """Chạy migrations ở 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True, # Cho phép quét đa schema
            include_name=include_name,
            version_table_schema="public"
        )

        with context.begin_transaction():
            context.run_migrations()
            custom_logger.info("Hoàn tất chạy migration (Online mode)")

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()