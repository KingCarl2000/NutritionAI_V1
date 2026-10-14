from sqlalchemy.orm import sessionmaker, declarative_base

# Import engine từ dự án của bạn (sử dụng connection pool đã cấu hình)
try:
    from src.postgres.core.connection import get_engine
    engine = get_engine()
except ImportError:
    engine = None

# Import Custom Logger của bạn
from src.nutrition_core.logging.logger import logger

# ==========================================
# 1. CẤU HÌNH CORE SESSION & BASE METADATA
# ==========================================
# Khởi tạo nhà máy tạo Session (SessionLocal)
# autocommit=False, autoflush=False để kiểm soát giao dịch an toàn (ACID)
if engine:
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    SessionLocal = None
    logger.warning("Engine chưa được khởi tạo. SessionLocal hiện đang là None.")

# Base class gốc để theo dõi toàn bộ Table (Models sẽ kế thừa class này)
Base = declarative_base()

# ==========================================
# 2. DEPENDENCY / CONTEXT MANAGER CHO CRUD
# ==========================================
def get_db():
    """
    Dependency dùng để inject session vào FastAPI hoặc các module Service/CRUD.
    Đảm bảo an toàn vòng đời transaction: tự động commit/rollback và luôn đóng kết nối.
    """
    if not SessionLocal:
        raise RuntimeError("Database session factory (SessionLocal) is not initialized.")
        
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Lỗi transaction trên Database, đã tự động rollback: {e}")
        raise
    finally:
        db.close()