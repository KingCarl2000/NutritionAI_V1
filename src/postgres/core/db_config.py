import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

@dataclass
class DatabaseConfig:
    """Lưu trữ cấu hình kết nối PostgreSQL và các tham số cho Python App."""
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", 5432))
    dbname: str = os.getenv("POSTGRES_DB", "nutrition_ai")
    user: str = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "mysecretpassword")
    
    # Giữ lại biến này vì nó phục vụ logic Python (không phải của Postgres Server)
    training_chunk_size: int = int(os.getenv("POSTGRES_TRAINING_CHUNK_SIZE", 10000))

    def get_connection_string(self) -> str:
        """Tạo chuỗi kết nối chuẩn DSN cho psycopg."""
        return f"dbname={self.dbname} user={self.user} password={self.password} host={self.host} port={self.port}"

config = DatabaseConfig()