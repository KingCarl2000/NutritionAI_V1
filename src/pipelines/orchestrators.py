import os
import pandas as pd
from typing import Dict, Any

from src.nutrition_core.logging.logger import logger
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker
from src.pipelines.ml_pipelines.calories_pred.data_ingest import CaloriesDataIngestion
from src.pipelines.ml_pipelines.calories_pred.data_val import run_validation_pipeline
from src.pipelines.ml_pipelines.calories_pred.eval_drift import CaloriesDriftEvaluator

class CaloriesPipelineOrchestrator:
    def __init__(self, ingest_config: dict, drift_config: dict):
        """
        Khởi tạo Orchestrator cho Data Pipeline (Ingestion -> Validation -> Drift)
        """
        self.ingest_config = ingest_config
        self.drift_config = drift_config
        
        # Khởi tạo MLflow tracker chung cho toàn bộ pipeline để duy trì cùng 1 run_id
        self.tracker = MLOpsTracker(experiment_name="calories_pred")
        
    def run_pipeline(self):
        logger.info("BẮT ĐẦU CHẠY PIPELINE: INGESTION -> VALIDATION -> DRIFT")
        
        # Bắt đầu MLflow run
        self.tracker.start_run(run_name="data_pipeline_orchestration")
        
        try:
            # ==========================================
            # 1. DATA INGESTION
            # ==========================================
            logger.info("--- GIAI ĐOẠN 1: DATA INGESTION ---")
            ingestor = CaloriesDataIngestion(config=self.ingest_config, tracker=self.tracker)
            
            # Trích xuất dữ liệu từ PostgreSQL
            raw_df = ingestor.extract_data_from_postgresql()
            
            # (Giả định module ingestor có hàm split_and_save_data trả về các dataframe hoặc đường dẫn)
            # Ở đây chúng ta chia train/test/val theo như thiết kế kiến trúc
            train_df, test_df, val_df = ingestor.split_and_save_data(raw_df) 
            logger.info("Data Ingestion hoàn tất.")

            # ==========================================
            # 2. DATA VALIDATION (PANDERA)
            # ==========================================
            logger.info("--- GIAI ĐOẠN 2: DATA VALIDATION ---")
            
            # Validate tập Train
            valid_train_df = run_validation_pipeline(
                df=train_df, 
                mode="train", 
                tracker=self.tracker
            )
            
            # Validate tập Test (Dùng schema input)
            valid_test_df = run_validation_pipeline(
                df=test_df, 
                mode="test", 
                tracker=self.tracker
            )

            # Validate tập Validation (Dùng schema input)
            valid_val_df = run_validation_pipeline(
                df=val_df, 
                mode="val", 
                tracker=self.tracker
            )
            logger.info("Data Validation hoàn tất.")

            # ==========================================
            # 3. DATA DRIFT DETECTION (EVIDENTLY AI)
            # ==========================================
            logger.info("--- GIAI ĐOẠN 3: DATA DRIFT DETECTION ---")
            drift_evaluator = CaloriesDriftEvaluator(config=self.drift_config, tracker=self.tracker)
            
            # Đánh giá Drift giữa tập Train (Reference) và tập Test (Current)
            is_drifted = drift_evaluator.evaluate_drift(
                reference_df=valid_train_df,
                current_df=valid_test_df
            )
            
            if is_drifted:
                logger.warning("CẢNH BÁO: Phát hiện Data Drift! Có thể cần trigger retrain.")
                self.tracker.log_params_and_metrics(metrics={"data_drift_detected": 1})
            else:
                logger.info("Không phát hiện Data Drift nghiêm trọng.")
                self.tracker.log_params_and_metrics(metrics={"data_drift_detected": 0})
                
            logger.info("Data Drift Detection hoàn tất.")
            logger.info("PIPELINE HOÀN TẤT THÀNH CÔNG.")
            
        except Exception as e:
            logger.error(f"Pipeline thất bại với lỗi: {str(e)}")
            self.tracker.log_error_and_tag(error_msg=str(e))
            raise e
        finally:
            # Tải log hệ thống lên MLflow và đóng run
            self.tracker.upload_system_logs()
            self.tracker.end_run()

if __name__ == "__main__":
    import yaml
    
    # Load configs
    with open("config/pipelines/ml_calories/data_ingestion.yaml", "r") as f:
        ingest_config = yaml.safe_load(f)
        
    with open("config/pipelines/ml_calories/evidently_profile.yaml", "r") as f:
        drift_config = yaml.safe_load(f)
        
    # Chạy orchestrator
    orchestrator = CaloriesPipelineOrchestrator(
        ingest_config=ingest_config,
        drift_config=drift_config
    )
    orchestrator.run_pipeline()