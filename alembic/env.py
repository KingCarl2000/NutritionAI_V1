import logging
import sys
import os
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool
from alembic import context

# 1. Thêm root path của project vào sys.path để import từ 'src'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# 2. Import Custom Logger của dự án
try:
    from src.nutrition_core.logging.logger import logger as custom_logger
except ImportError:
    custom_logger = logging.getLogger("alembic.env")

# 3. NẠP BASE VÀ MODELS ĐỂ ALEMBIC THU THẬP METADATA
try:
    from src.nutrition_core.database.session import Base
    # Bắt buộc import module models để thực thi việc nạp model (CaloriesRaw) vào Base.metadata
    from src.nutrition_core.database import models
    
    target_metadata = Base.metadata
    custom_logger.info("[Alembic] Nạp thành công target_metadata từ Base.metadata")
except ImportError as e:
    target_metadata = None
    custom_logger.error(f"[Alembic] Không thể nạp Base/Models từ database module. Lỗi: {e}")

# Đọc file cấu hình alembic.ini
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 4. Tự động nạp URL Database từ src.postgres.core.connection
try:
    from src.postgres.core.connection import get_engine
    engine = get_engine()
    # Ghi đè URL trong alembic.ini bằng URL thực tế từ connection engine
    config.set_main_option("sqlalchemy.url", str(engine.url))
except Exception as e:
    custom_logger.warning(f"[Alembic] Cảnh báo: Không thể nạp database URL từ connection.py. Lỗi: {e}")

# 5. Cấu hình quản lý Multi-Schema (Hỗ trợ schema raw, public, staging)
TARGET_SCHEMAS = ["public", "raw", "staging"]

def include_name(name, type_, parent_names):
    """Giúp Alembic theo dõi và tự động phát hiện thay đổi trên nhiều schema khác nhau."""
    if type_ == "schema":
        return name in TARGET_SCHEMAS
    return True

def run_migrations_offline() -> None:
    """Chạy migration ở chế độ 'offline'."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Chạy migration ở chế độ 'online'."""
    try:
        connectable = get_engine()
    except Exception:
        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            include_name=include_name,
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()