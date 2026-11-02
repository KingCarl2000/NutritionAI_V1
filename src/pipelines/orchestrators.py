import os
import yaml
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker
from src.pipelines.ml_pipelines.calories_pred.data_ingest import CaloriesDataIngestion
from src.pipelines.ml_pipelines.calories_pred.data_val import CaloriesDataValidationPipeline

def load_yaml_config(path: str) -> dict:
    """Đọc file cấu hình YAML."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Không tìm thấy file cấu hình tại {path}")
        raise

class CaloriesOrchestrator:
    """
    Nhạc trưởng điều phối luồng MLOps cho Calories Prediction.
    Thực thi tuần tự các giai đoạn và chia sẻ chung MLOpsTracker.
    """
    def __init__(self, ingest_config_path: str, val_config_path: str):
        logger.info("Khởi tạo CaloriesOrchestrator...")
        self.ingest_config = load_yaml_config(ingest_config_path)
        self.val_config = load_yaml_config(val_config_path)
        
        # Khởi tạo tracker duy nhất để đồng bộ run_id qua các bước
        self.tracker = MLOpsTracker(experiment_name="calories_pred_pipeline")

    def run_initial_stages(self):
        """Chạy kiểm thử 2 giai đoạn đầu: Data Ingestion & Data Validation"""
        logger.info("BẮT ĐẦU KIỂM THỬ: Ingestion -> Validation")
        
        with self.tracker.start_run(run_name="test_ingest_val_pipeline"):
            try:
                # ==========================================
                # GIAI ĐOẠN 1: DATA INGESTION
                # ==========================================
                logger.info("=== GIAI ĐOẠN 1: DATA INGESTION (PostgreSQL) ===")
                ingestion_pipeline = CaloriesDataIngestion(
                    config=self.ingest_config, 
                    tracker=self.tracker
                )
                # Thực thi trích xuất và lưu DVC Artifacts (train/test/val)
                ingestion_pipeline.extract_data_from_postgresql()
                
                # ==========================================
                # GIAI ĐOẠN 2: DATA VALIDATION & DRIFT
                # ==========================================
                logger.info("=== GIAI ĐOẠN 2: DATA VALIDATION & DRIFT CHECK ===")
                validation_pipeline = CaloriesDataValidationPipeline(
                    # Truyền config hoặc các tham số cần thiết (tuỳ thuộc vào `__init__` hiện tại của bạn)
                    # config=self.val_config, 
                    # tracker=self.tracker 
                )
                # class ValidationPipeline sẽ tự động nạp Artifacts do Ingestion tạo ra và chạy
                validation_pipeline.run()
                
                logger.info("KIỂM THỬ THÀNH CÔNG: Hoàn thành Ingestion và Validation.")
                
            except Exception as e:
                logger.error(f"Lỗi nghiêm trọng trong quá trình Orchestration: {str(e)}")
                self.tracker.log_error_and_tag(error=e, stage="orchestration_test")
                raise

if __name__ == "__main__":
    # Thay đổi đường dẫn config cho phù hợp với cấu trúc thư mục thực tế của bạn
    INGEST_YAML = "config/pipelines/ml_calories/data_ingestion.yaml"
    VAL_YAML = "config/pipelines/ml_calories/data_validation.yaml"
    
    orchestrator = CaloriesOrchestrator(
        ingest_config_path=INGEST_YAML,
        val_config_path=VAL_YAML
    )
    
    # Kích hoạt chạy kiểm thử 2 bước
    orchestrator.run_initial_stages()
