import os
import yaml
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker
from src.pipelines.ml_pipelines.calories_pred.data_ingest import CaloriesDataIngestion

# Import đúng hàm từ data_val.py thay vì class
from src.pipelines.ml_pipelines.calories_pred.data_val import run_validation_pipeline
from src.pipelines.ml_pipelines.calories_pred.eval_drift import CaloriesDriftEvaluator

def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class CaloriesOrchestrator:
    def __init__(self):
        logger.info("Khởi tạo CaloriesOrchestrator...")
        self.ingest_config = load_yaml_config("config/pipelines/ml_calories/data_ingestion.yaml")
        self.val_config = load_yaml_config("config/pipelines/ml_calories/data_val.yaml")
        self.drift_config = "config/pipelines/ml_calories/evidently_profile.yaml"
        
        # 1. Khởi tạo Tracker duy nhất
        self.tracker = MLOpsTracker(experiment_name="calories_pred")
        
        # 2. Khởi tạo các module dạng hướng đối tượng
        self.ingestion = CaloriesDataIngestion(config=self.ingest_config, tracker=self.tracker)
        self.drift_evaluator = CaloriesDriftEvaluator(evidently_config_path=self.drift_config)

    def run_pipeline(self):
        try:
            self.tracker.start_run()
            logger.info("================ BẮT ĐẦU PIPELINE ==================")

            # BƯỚC 1: INGESTION
            logger.info(">>> THỰC THI DATA INGESTION")
            # Giả định ingestion.run() trả về train, val, test
            train_df, val_df, test_df = self.ingestion.run()

            # BƯỚC 2: VALIDATION
            logger.info(">>> THỰC THI DATA VALIDATION")
            # Gọi hàm run_validation_pipeline cho từng tập dữ liệu, truyền chung tracker
            valid_train_df = run_validation_pipeline(
                df=train_df, 
                config=self.val_config, 
                tracker=self.tracker, 
                dataset_name="train"
            )
            valid_val_df = run_validation_pipeline(
                df=val_df, 
                config=self.val_config, 
                tracker=self.tracker, 
                dataset_name="val"
            )
            valid_test_df = run_validation_pipeline(
                df=test_df, 
                config=self.val_config, 
                tracker=self.tracker, 
                dataset_name="test"
            )

            # BƯỚC 3: DATA DRIFT
            logger.info(">>> THỰC THI DATA DRIFT EVALUATION")
            self.drift_evaluator.evaluate(
                reference_df=valid_train_df,
                current_df=valid_test_df,
                tracker=self.tracker 
            )

            logger.info("================ HOÀN THÀNH ==================")
            
        except Exception as e:
            logger.error(f"Pipeline thất bại: {str(e)}")
            raise
        finally:
            self.tracker.end_run()

if __name__ == "__main__":
    orchestrator = CaloriesOrchestrator()
    orchestrator.run_pipeline()