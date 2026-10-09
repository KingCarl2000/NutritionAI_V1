from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

# Tận dụng custom logger đã thiết lập trong dự án
from src.nutrition_core.logging.logger import logger

# Tái sử dụng engine từ hạ tầng PostgreSQL hiện tại
from src.postgres.core.connection import get_engine

# 1. Lấy Engine (Đã được cấu hình connection pool và timeouts bên connection.py)
engine = get_engine()

# 2. Khởi tạo Session Factory
# autocommit=False: Bắt buộc gọi db.commit() để tránh ghi nhầm dữ liệu lỗi vào DB.
# autoflush=False: Không tự động gửi query xuống DB cho đến khi thực sự cần.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 3. Base class chuẩn bị sẵn cho các bảng (khi bạn tạo bảng theo dõi Model/Metrics sau này)
Base = declarative_base()

# 4. Trình quản lý phiên an toàn (Dependency Generator)
def get_db():
    """
    Cấp phát một phiên kết nối CSDL và đảm bảo thu hồi (close) ngay cả khi có lỗi.
    Sử dụng cho các script MLOps để đọc/ghi dữ liệu bảng hoặc metadata.
    """
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        # Sử dụng logger của dự án để ghi lỗi vào file error_logs
        logger.error(f"Lỗi truy vấn Database (SQLAlchemyError): {str(e)}")
        db.rollback()  # Xóa bỏ các thay đổi chưa hoàn tất để tránh treo giao dịch
        raise
    except Exception as e:
        logger.error(f"Lỗi hệ thống trong phiên làm việc DB: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()  # Bắt buộc trả kết nối về Pool