import os
import yaml
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker
from src.pipelines.ml_pipelines.calories_pred.data_ingest import CaloriesDataIngestion
from src.pipelines.ml_pipelines.calories_pred.data_val import CaloriesDataValidationPipeline
from src.pipelines.ml_pipelines.calories_pred.eval_drift import CaloriesDriftEvaluator

def load_yaml_config(path: str) -> dict:
    """Đọc file cấu hình YAML."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Không tìm thấy file cấu hình tại {path}")
        raise

class CaloriesOrchestrator:
    def __init__(self, 
                 ingest_config_path: str = "config/pipelines/ml_calories/data_ingestion.yaml",
                 val_config_path: str = "config/pipelines/ml_calories/data_val.yaml",
                 drift_config_path: str = "config/pipelines/ml_calories/evidently_profile.yaml"):
        
        logger.info("Khởi tạo CaloriesOrchestrator...")
        self.ingest_config = load_yaml_config(ingest_config_path)
        self.val_config = load_yaml_config(val_config_path)
        
        # 1. Khởi tạo MLOpsTracker DUY NHẤT để dùng chung cho toàn bộ pipeline (cùng 1 run_id)
        self.tracker = MLOpsTracker(experiment_name="calories_pred")
        
        # 2. Khởi tạo các module
        self.ingestion = CaloriesDataIngestion(config=self.ingest_config, tracker=self.tracker)
        self.validation = CaloriesDataValidationPipeline(config=self.val_config, tracker=self.tracker)
        self.drift_evaluator = CaloriesDriftEvaluator(evidently_config_path=drift_config_path)

    def run_pipeline(self):
        try:
            # Bắt đầu MLflow Run
            self.tracker.start_run()
            logger.info("================ BẮT ĐẦU PIPELINE CALORIES PRED ==================")

            # BƯỚC 1: DATA INGESTION
            logger.info(">>> BƯỚC 1: THỰC THI DATA INGESTION")
            # Ingestion trích xuất dữ liệu từ Postgres, có thể trả về các dataframe train, val, test
            train_df, val_df, test_df = self.ingestion.run()

            # BƯỚC 2: DATA VALIDATION
            logger.info(">>> BƯỚC 2: THỰC THI DATA VALIDATION")
            # Truyền dữ liệu thô vào bộ Validation bằng Pandera để làm sạch và cách ly lỗi
            valid_train_df, valid_val_df, valid_test_df = self.validation.run(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df
            )

            # BƯỚC 3: DATA DRIFT EVALUATION
            logger.info(">>> BƯỚC 3: THỰC THI DATA DRIFT EVALUATION")
            # So sánh phân phối của dữ liệu Test (Current) so với dữ liệu Train (Reference)
            # Truyền thêm tracker vào để ghi nhận evidently HTML report lên MLflow Artifacts
            self.drift_evaluator.evaluate(
                reference_df=valid_train_df,
                current_df=valid_test_df,
                tracker=self.tracker 
            )

            logger.info("================ HOÀN THÀNH PIPELINE ==================")
            
        except Exception as e:
            logger.error(f"Pipeline thất bại do lỗi: {str(e)}")
            raise
        finally:
            # Đảm bảo MLflow Run luôn được đóng dù xảy ra lỗi
            self.tracker.end_run()

if __name__ == "__main__":
    orchestrator = CaloriesOrchestrator()
    orchestrator.run_pipeline()