import logging
import sys
import os
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# 1. Thêm root path của project vào sys.path để có thể import từ thư mục 'src'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# 2. Import custom logger từ core của dự án (dựa trên cấu trúc tab của bạn)
try:
    from src.nutrition_core.logging.logger import logger as custom_logger
except ImportError:
    custom_logger = logging.getLogger("alembic.env")

# 3. Import Base metadata từ project của bạn để hỗ trợ tính năng 'autogenerate'
# Lưu ý: Sửa lại đường dẫn import này trỏ đúng tới file chứa declarative_base() của bạn
try:
    from src.postgres.core.db_config import Base  # Giả định Base nằm ở đây
    target_metadata = Base.metadata
except ImportError:
    target_metadata = None
    custom_logger.warning("Không thể import Base metadata. Autogenerate có thể không hoạt động chính xác.")

config = context.config

# Thiết lập logging cơ bản của Alembic (vẫn giữ nguyên để tương thích file .ini)
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 4. Ghi đè sqlalchemy.url từ cấu hình hệ thống nếu cần (Bảo mật hơn file alembic.ini)
# try:
#     from src.postgres.core.connection import get_database_url
#     config.set_main_option("sqlalchemy.url", get_database_url())
# except ImportError:
#     pass

# 5. Cấu hình quản lý Multi-Schema (Dựa trên sơ đồ Data Ingestion)
# Sơ đồ yêu cầu quản lý 'raw Schema' và 'public/staging Schema'
TARGET_SCHEMAS = ["public", "raw", "staging"]

def include_name(name, type_, parent_names):
    """
    Hàm này giúp Alembic nhận diện và theo dõi các thay đổi trên nhiều schemas cụ thể.
    """
    if type_ == "schema":
        return name in TARGET_SCHEMAS
    else:
        return True

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True, # Bật tính năng quét nhiều schema
        include_name=include_name,
        version_table_schema="public" # Bảng alembic_version sẽ lưu ở public
    )

    with context.begin_transaction():
        context.run_migrations()
        custom_logger.info("Hoàn tất chạy migration (Offline mode)")

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True, # Bật tính năng quét nhiều schema
            include_name=include_name,
            version_table_schema="public" # Bảng alembic_version sẽ lưu ở public
        )

        with context.begin_transaction():
            context.run_migrations()
            custom_logger.info("Hoàn tất chạy migration (Online mode)")

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()