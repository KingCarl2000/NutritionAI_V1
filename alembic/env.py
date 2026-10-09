# alembic/env.py
import os
import sys
import logging
from alembic import context

# 1. Thêm root project vào sys.path để import các module từ src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. Import Engine, Base, Models và Logger từ dự án
from src.postgres.core.connection import get_engine
from src.nutrition_core.database.session import Base
import src.nutrition_core.database.models  # noqa (Bắt buộc import để Alembic đọc metadata)
from src.nutrition_core.logging.logger import logger as app_logger

# 3. Chuyển hướng Logging của Alembic & SQLAlchemy vào hệ thống Log của dự án
alembic_logger = logging.getLogger("alembic")
sqlalchemy_logger = logging.getLogger("sqlalchemy.engine")

# Xóa các handler mặc định (nếu có) và gắn các handler từ logger dự án (oper, warm, error)
alembic_logger.handlers.clear()
sqlalchemy_logger.handlers.clear()

for handler in app_logger.handlers:
    alembic_logger.addHandler(handler)
    sqlalchemy_logger.addHandler(handler)

alembic_logger.setLevel(logging.INFO)
sqlalchemy_logger.setLevel(logging.WARNING) # Đổi thành INFO nếu bạn muốn in cả câu lệnh SQL ra file log

# 4. Gán target_metadata
target_metadata = Base.metadata

def include_name(name, type_, parent_names):
    """
    Lọc schema: Chỉ quét bảng trong schema 'public' hoặc 'nutrition_ai'
    Bỏ qua các schema hệ thống của PostgreSQL (pg_catalog, information_schema)
    """
    if type_ == "schema":
        return name in [None, "public", "core", "raw", "staging"]
    return True

def run_migrations_offline() -> None:
    """Chạy migrations ở chế độ 'offline' (Sinh script SQL)."""
    app_logger.info("Chạy Alembic migrations ở chế độ OFFLINE")
    engine = get_engine()
    
    context.configure(
        url=engine.url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        transactional_ddl=True 
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Chạy migrations ở chế độ 'online' (Kết nối trực tiếp vào DB)."""
    app_logger.info("Chạy Alembic migrations ở chế độ ONLINE")
    
    # Lấy engine đã được khởi tạo sẵn (Singleton) kèm pool cấu hình từ db_config.py
    connectable = get_engine()

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            
            # --- CẤU HÌNH TỐI ƯU CHO POSTGRESQL ---
            include_schemas=True,           # Hỗ trợ đa Schema
            include_name=include_name,      # Loại trừ schema hệ thống
            compare_type=True,              # Tracking kiểu dữ liệu Postgres (JSONB, ARRAY, ENUM...)
            compare_server_default=True,    # Tracking giá trị DEFAULT
            transactional_ddl=True,         # Gói DDL trong transaction (Tự động Rollback nếu lỗi)
            dialect_name="postgresql"       # Ép kiểu dialect rõ ràng
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()