import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Tải các biến từ file .env vào biến môi trường hệ thống
load_dotenv()

@dataclass
class DatabaseConfig:
    """Lưu trữ cấu hình kết nối PostgreSQL và các tham số tối ưu hóa cho MLOps."""
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
        """Tạo chuỗi kết nối chuẩn DSN cho psycopg3."""
        return f"dbname={self.dbname} user={self.user} password={self.password} host={self.host} port={self.port}"

config = DatabaseConfig()