import os
import yaml
import pandas as pd

from src.nutrition_core.logging.logger import logger
from src.nutrition_core.pandera.evidently_engine import EvidentlyDriftEngine
from src.nutrition_core.database.session import get_engine
from src.data_io.base_extractor import BasePostgresExtractor

class CaloriesExtractor(BasePostgresExtractor):
    def __init__(self, engine, schema: str, table: str):
        self.engine = engine
        self.schema = schema
        self.table = table
        
    def extract(self) -> pd.DataFrame:
        """Trích xuất và phẳng hóa dữ liệu JSON/JSONB nếu có."""
        query = f"SELECT * FROM {self.schema}.{self.table};"
        df = pd.read_sql(query, con=self.engine)
        
        if hasattr(self, '_convert_data_types'):
            df = self._convert_data_types(df)
        return df

def load_yaml_config(config_path: str) -> dict:
    """Hàm nạp cấu hình dùng chung."""
    if not os.path.exists(config_path):
        logger.warning(f"Không tìm thấy cấu hình: {config_path}")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def fetch_current_data_from_postgres(schema: str, table: str) -> pd.DataFrame:
    """Lấy dữ liệu từ PostgreSQL dựa trên Schema & Table (không hardcode URL)."""
    logger.info(f"Đang kết nối PostgreSQL để lấy dữ liệu đánh giá từ {schema}.{table}...")
    
    # get_engine() tự động xử lý URL từ môi trường hoặc secret config
    engine = get_engine()
    extractor = CaloriesExtractor(engine=engine, schema=schema, table=table)
    df = extractor.extract() 
    
    logger.info(f"📥 Đã lấy thành công {len(df)} bản ghi từ PostgreSQL ({schema}.{table})")
    return df

def execute_data_drift_detection(
    profile_path: str = "config/pipelines/ml_calories/evidently_profile.yaml",
    global_path: str = "config/pipelines/global.yaml"
):
    logger.info("--- BẮT ĐẦU ĐÁNH GIÁ DATA DRIFT ---")
    
    # 1. Đọc linh động cấu hình từ YAML
    profile_cfg = load_yaml_config(profile_path)
    global_cfg = load_yaml_config(global_path)
    
    paths_cfg = profile_cfg.get("paths", {})
    current_data_cfg = profile_cfg.get("current_data", {})
    
    # Lấy ngưỡng thống kê toàn cục từ global.yaml
    global_drift_cfg = global_cfg.get("mlops_thresholds", {}).get("data_drift", {})
    p_value = global_drift_cfg.get("p_value_alpha", 0.05)
    
    # 2. Mapping các thông số
    reference_path = paths_cfg.get("reference_data")
    artifact_dir = paths_cfg.get("artifact_dir")
    project_name = paths_cfg.get("project_name")
    
    eval_schema = current_data_cfg.get("db_schema")
    eval_table = current_data_cfg.get("table_name")
    
    if not all([reference_path, artifact_dir, project_name, eval_schema, eval_table]):
         raise ValueError("Thiếu cấu hình thiết yếu trong evidently_profile.yaml!")
         
    # Cập nhật ngưỡng p_value toàn cục vào hồ sơ evidently
    if "data_drift" not in profile_cfg:
        profile_cfg["data_drift"] = {}
    profile_cfg["data_drift"]["num_stattest_threshold"] = p_value
    profile_cfg["data_drift"]["cat_stattest_threshold"] = p_value

    # 3. Trích xuất dữ liệu tự động
    current_df = fetch_current_data_from_postgres(schema=eval_schema, table=eval_table)
    
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Không tìm thấy reference_data tại: {reference_path}")
    reference_df = pd.read_csv(reference_path)
    
    # 4. Thực thi Engine
    drift_engine = EvidentlyDriftEngine(
        artifact_dir=artifact_dir, 
        project_name=project_name
    )
    
    drift_engine.run_drift_detection(
        current_df=current_df, 
        reference_df=reference_df,
        profile_config=profile_cfg 
    )
    logger.info("--- HOÀN TẤT ĐÁNH GIÁ DATA DRIFT ---")

if __name__ == "__main__":
    execute_data_drift_detection()