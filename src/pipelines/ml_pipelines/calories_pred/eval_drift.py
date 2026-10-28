import os
import sys
import yaml
import pandas as pd

# Import Logger và EvidentlyDriftEngine đã được định nghĩa trong hệ thống
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.pandera.evidently_engine import EvidentlyDriftEngine

# Tùy thuộc vào kiến trúc, bạn import class kết nối DB (BasePostgresExtractor) hoặc get_engine
# Giả định dùng session để lấy sqlalchemy engine hoặc BasePostgresExtractor theo docs/src.md
try:
    from src.nutrition_core.database.session import get_engine
except ImportError:
    # Nếu chưa có session.py, import thư viện để tạo engine tạm
    from sqlalchemy import create_engine

def load_yaml_config(config_path: str) -> dict:
    """Đọc tệp cấu hình YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy cấu hình: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def fetch_current_data_from_postgres(
    db_config_path: str = "config/datasets/calories.yaml"
) -> pd.DataFrame:
    """
    Trích xuất dữ liệu current từ PostgreSQL.
    Theo image.png và calories.yaml, data nằm ở: DB nutrition_ai, schema raw, table calories.
    """
    logger.info("Đang kết nối PostgreSQL để lấy dữ liệu hiện tại (Current Data)...")
    
    # Lấy thông tin table/schema từ cấu hình dataset
    dataset_cfg = load_yaml_config(db_config_path).get("calories", {})
    schema = dataset_cfg.get("db_schema", "raw")
    table = dataset_cfg.get("table_name", "calories")
    
    try:
        # Sử dụng engine dùng chung của dự án (khuyến nghị)
        engine = get_engine()
    except NameError:
        # Thay thế bằng URI kết nối thực tế của bạn nếu get_engine chưa sẵn sàng
        db_url = "postgresql://username:password@localhost:5432/nutrition_ai"
        engine = create_engine(db_url)
        
    query = f"SELECT * FROM {schema}.{table};"
    
    # Thực thi truy vấn lấy data bằng pandas
    df = pd.read_sql(query, con=engine)
    logger.info(f"📥 Đã lấy thành công {len(df)} bản ghi từ PostgreSQL ({schema}.{table})")
    
    return df

def execute_data_drift_detection(
    reference_path: str = "Artifacts/data/reference_data.csv",
    evidently_profile_path: str = "config/pipelines/ml_calories/evidently_profile.yaml",
    artifact_dir: str = "Artifacts/reports/ml_calories/evidently",
    project_name: str = "ml_calories"
):
    """Thực thi kiểm tra Data Drift dùng EvidentlyDriftEngine và cấu hình profile riêng."""
    logger.info("--- BẮT ĐẦU ĐÁNH GIÁ DATA DRIFT TỪ POSTGRES ---")

    # 1. Đọc cấu hình evidently_profile.yaml của pipeline hiện tại
    evidently_cfg = load_yaml_config(evidently_profile_path)
    logger.info(f"Đã nạp cấu hình evidently profile từ: {evidently_profile_path}")
    
    # Lấy ra các tham số column mapping từ cấu hình (thay đổi key tùy theo file yaml của bạn)
    column_mapping_cfg = evidently_cfg.get("column_mapping", {})

    # 2. Đọc dữ liệu Reference từ tệp (DVC artifact)
    if not os.path.exists(reference_path):
        logger.error(f"Không tìm thấy reference data: {reference_path}")
        raise FileNotFoundError(f"Missing reference data: {reference_path}")
    
    logger.info(f"📥 Đang tải tập dữ liệu tham chiếu (Reference): {reference_path}")
    reference_df = pd.read_csv(reference_path)

    # 3. Đọc dữ liệu Current trực tiếp từ DB PostgreSQL thay vì CSV
    current_df = fetch_current_data_from_postgres()

    # 4. Khởi tạo EvidentlyDriftEngine
    drift_engine = EvidentlyDriftEngine(
        artifact_dir=artifact_dir,
        project_name=project_name
    )

    # 5. Thực thi phát hiện Drift thông qua Engine tích hợp sẵn
    logger.info("⚙️ Đang kích hoạt EvidentlyDriftEngine.run_drift_detection()...")
    
    # Lưu ý: Phương thức run_drift_detection đã bao gồm việc lưu HTML, JSON và đẩy lên MLflow
    # nếu mlops_tracker được tích hợp.
    result = drift_engine.run_drift_detection(
        current_df=current_df,
        reference_df=reference_df,
        column_mapping_cfg=column_mapping_cfg
    )
    
    logger.info("✅ Hoàn tất đánh giá Data Drift Pipeline.")
    return result

if __name__ == "__main__":
    try:
        execute_data_drift_detection()
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình chạy eval_drift.py: {e}")
        sys.exit(1)