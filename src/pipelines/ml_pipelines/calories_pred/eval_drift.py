import os
import sys
import yaml
import pandas as pd

# Import chuẩn theo kiến trúc từ docs/src.md
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.pandera.evidently_engine import EvidentlyDriftEngine
from src.nutrition_core.database.session import get_engine
from src.data_io.base_extractor import BasePostgresExtractor

# Vì BasePostgresExtractor là Abstract Base Class (ABC) theo docs/src.md,
# ta định nghĩa một lớp con cụ thể để thực thi cho bảng calories.
class CaloriesExtractor(BasePostgresExtractor):
    def __init__(self, engine, schema: str, table: str):
        self.engine = engine
        self.schema = schema
        self.table = table
        
    def extract(self) -> pd.DataFrame:
        """Trích xuất và phẳng hóa dữ liệu JSON/JSONB nếu có."""
        query = f"SELECT * FROM {self.schema}.{self.table};"
        df = pd.read_sql(query, con=self.engine)
        
        # Tận dụng hàm _convert_data_types() từ class cha để flatten JSON (như thiết kế trong src.md)
        if hasattr(self, '_convert_data_types'):
            df = self._convert_data_types(df)
        return df

def load_yaml_config(config_path: str) -> dict:
    """Đọc tệp cấu hình YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy cấu hình: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def fetch_current_data_from_postgres(schema: str, table: str) -> pd.DataFrame:
    """
    Trích xuất dữ liệu Current từ PostgreSQL thông qua session chuẩn và Extractor.
    """
    logger.info(f"Đang kết nối PostgreSQL để lấy dữ liệu đánh giá từ {schema}.{table}...")
    
    # 1. Sử dụng engine chuẩn từ session.py (Không hardcode credentials)
    engine = get_engine()
    
    # 2. Sử dụng Extractor để đảm bảo dữ liệu qua các bước tiền xử lý đồng nhất
    extractor = CaloriesExtractor(engine=engine, schema=schema, table=table)
    df = extractor.extract() 
    
    logger.info(f"📥 Đã lấy thành công {len(df)} bản ghi từ PostgreSQL ({schema}.{table})")
    return df

def execute_data_drift_detection(
    profile_path: str = "config/pipelines/ml_calories/evidently_profile.yaml"
):
    """Thực thi kiểm tra Data Drift dùng EvidentlyDriftEngine."""
    logger.info("--- BẮT ĐẦU ĐÁNH GIÁ DATA DRIFT ---")
    
    # 1. Đọc cấu hình evidently_profile.yaml
    profile_cfg = load_yaml_config(profile_path)
    paths_cfg = profile_cfg.get("paths", {})
    current_data_cfg = profile_cfg.get("current_data", {})
    
    logger.info(f"Đã nạp cấu hình evidently profile từ: {profile_path}")
    
    # 2. Khởi tạo đường dẫn từ YAML
    reference_path = paths_cfg.get("reference_data", "Artifacts/data/ml_calories/processed/train.csv")
    artifact_dir = paths_cfg.get("artifact_dir", "Artifacts/reports/ml_calories/evidently")
    project_name = paths_cfg.get("project_name", "ml_calories")
    
    # 3. Lấy thông tin Schema và Table cho Current Data
    eval_schema = current_data_cfg.get("db_schema", "raw")
    eval_table = current_data_cfg.get("table_name", "calories")
    
    # 4. Trích xuất dữ liệu Current và đọc Reference data
    current_df = fetch_current_data_from_postgres(schema=eval_schema, table=eval_table)
    
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Không tìm thấy file reference_data tại: {reference_path}")
    reference_df = pd.read_csv(reference_path)
    
    # 5. Khởi tạo Engine và chạy Drift Detection
    drift_engine = EvidentlyDriftEngine(
        artifact_dir=artifact_dir, 
        project_name=project_name
    )
    
    # Truyền toàn bộ profile_config để Evidently ánh xạ cột (column_mapping) và thresholds
    drift_engine.run_drift_detection(
        current_df=current_df, 
        reference_df=reference_df,
        profile_config=profile_cfg 
    )
    logger.info("--- HOÀN TẤT ĐÁNH GIÁ DATA DRIFT ---")

if __name__ == "__main__":
    # Điểm entry-point chạy script độc lập
    execute_data_drift_detection()