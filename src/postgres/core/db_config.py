import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# 1. Xác định đường dẫn thư mục gốc của dự án (Project Root)
# Tính từ vị trí file db_config.py này (src/postgres/core), lùi lại 3 cấp sẽ ra thư mục gốc
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 2. Chỉ định chính xác đường dẫn tới file .env ở thư mục gốc
ENV_PATH = PROJECT_ROOT / ".env"

# Tải các biến môi trường từ file .env được chỉ định (override=True để ưu tiên .env)
load_dotenv(dotenv_path=ENV_PATH, override=True)

@dataclass
class DatabaseConfig:
    """Lưu trữ cấu hình kết nối PostgreSQL và các tham số tối ưu hóa cho MLOps."""
    # Lấy thông tin từ .env, chỉ dùng mặc định làm phương án dự phòng
    host: str = os.getenv("PG_HOST", "localhost")
    port: int = int(os.getenv("PG_PORT", 5432))
    dbname: str = os.getenv("PG_DB", "nutrition_ai")
    user: str = os.getenv("PG_USER", "postgres")
    password: str = os.getenv("PG_PASSWORD", "password")
    
    # Cấu hình tối ưu hóa cho quá trình Bulk Load (ETL)
    # maintenance_work_mem: Tăng tốc độ tạo lại Index và Foreign Key (ví dụ: '2GB')
    maintenance_work_mem: str = os.getenv("PG_MAINTENANCE_WORK_MEM", "2GB")
    
    # Kích thước chunk khi stream dữ liệu huấn luyện (tránh OOM)
    training_chunk_size: int = int(os.getenv("PG_TRAINING_CHUNK_SIZE", 10000))

    def get_connection_string(self) -> str:
        """Tạo chuỗi kết nối chuẩn DSN cho psycopg."""
        return f"dbname={self.dbname} user={self.user} password={self.password} host={self.host} port={self.port}"

# Khởi tạo đối tượng config duy nhất (Singleton) để import vào các module khác
config = DatabaseConfig()