import os
import yaml
import pandas as pd
import pandera as pa
from pandera.errors import SchemaErrors

# Import schema chuẩn từ module config
from config.pipelines.ml_calories.calories_schema import (
    MlCaloriesInputSchema, 
    MlCaloriesTrainingSchema
)
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker

# Giả định module eval_drift đã được định nghĩa trong cùng thư mục
from src.pipelines.ml_pipelines.calories_pred.eval_drift import run_drift_detection

# =============================================================================
# MLOPS PIPELINE: PANDERA VALIDATION, QUARANTINE ROUTING & ALERTING
# =============================================================================

def run_validation_pipeline(
    df: pd.DataFrame, 
    mode: str = "train", 
    tracker: MLOpsTracker = None, 
    error_threshold: float = 0.05,
    quarantine_dir: str = "Artifacts/reports/ml_calories/validation"
) -> pd.DataFrame:
    """
    Thực thi Data Validation bằng Pandera, phân tách dữ liệu lỗi (Quarantine) và kích hoạt Alert.
    """
    logger.info(f"Bắt đầu quá trình Data Validation (Chế độ: {mode.upper()})...")
    schema = MlCaloriesTrainingSchema if mode == "train" else MlCaloriesInputSchema

    try:
        valid_df = schema.validate(df, lazy=True)
        logger.info(f"Validation thành công 100%. {len(valid_df)} bản ghi hợp lệ.")
        if tracker:
            tracker.log_params_and_metrics(metrics={f"val_{mode}_error_rate": 0.0})
        return valid_df
        
    except SchemaErrors as err:
        failure_indices = err.failure_cases['index'].dropna().unique()
        invalid_df = df.loc[failure_indices]
        valid_df = df.drop(index=failure_indices)
        
        total_records = len(df)
        invalid_count = len(invalid_df)
        error_rate = invalid_count / total_records
        
        logger.warning(
            f"Phát hiện {invalid_count}/{total_records} bản ghi vi phạm logic "
            f"(Tỷ lệ lỗi: {error_rate:.2%})."
        )
        
        if tracker:
            tracker.log_params_and_metrics(
                metrics={
                    f"val_{mode}_error_rate": error_rate,
                    f"val_{mode}_invalid_count": invalid_count
                }
            )
        
        os.makedirs(quarantine_dir, exist_ok=True)
        quarantine_data_path = os.path.join(quarantine_dir, f"quarantine_{mode}_data.csv")
        invalid_df.to_csv(quarantine_data_path, index=False)
        
        report_path = os.path.join(quarantine_dir, f"error_report_{mode}.csv")
        err.failure_cases.to_csv(report_path, index=False)
        
        logger.info(f"Đã lưu dữ liệu lỗi tại {quarantine_data_path} và báo cáo tại {report_path}")
        
        if error_rate > error_threshold:
            alert_msg = f"DATA ALERT: Tỷ lệ lỗi ({error_rate:.2%}) vượt ngưỡng cho phép ({error_threshold:.2%}). Dừng pipeline!"
            logger.error(alert_msg)
            raise ValueError(alert_msg)
            
        return valid_df

# =============================================================================
# DATA VALIDATION PIPELINE ORCHESTRATOR
# =============================================================================

class CaloriesDataValidationPipeline:
    """
    Điều phối toàn bộ Giai đoạn 2: Validation & Drift Check.
    Tích hợp Pandera, Evidently AI, DVC và MLflow.
    """
    def __init__(self, config_paths: dict):
        # Truy xuất cấu trúc pipeline_paths từ YAML
        self.pipeline_paths = config_paths.get("pipeline_paths", {})
        
        # Mapping các đường dẫn đầu ra và reference data
        self.ref_data_path = self.pipeline_paths.get("reference_data_path", "Artifacts/data/ml_calories/processed/train.csv")
        self.valid_train_path = self.pipeline_paths.get("valid_train_data", "Artifacts/data/ml_calories/processed/valid_train.csv")
        self.valid_test_path = self.pipeline_paths.get("valid_test_data", "Artifacts/data/ml_calories/processed/valid_test.csv")
        self.validation_report = self.pipeline_paths.get("validation_report", "Artifacts/reports/ml_calories/validation/validation_report.json")
        self.drift_report = self.pipeline_paths.get("drift_report", "Artifacts/reports/ml_calories/evidently/drift_report.html")
        
        # Trích xuất folder chứa các output
        self.output_dir = os.path.dirname(self.valid_train_path)
        self.quarantine_dir = os.path.dirname(self.validation_report)
        
        # Thư mục Ingestion (fallback an toàn do hiện tại file YAML chưa định nghĩa ingest_dir)
        self.ingest_dir = config_paths.get("ingest_dir", self.pipeline_paths.get("ingest_dir", "Artifacts/data_ingestion/calories"))
        
        self.tracker = MLOpsTracker(experiment_name="calories_data_validation")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.quarantine_dir, exist_ok=True)

    def run(self):
        """Thực thi luồng Ingested Artifacts -> Validation -> Drift Detection -> DVC Artifacts."""
        self.tracker.start_run()
        try:
            # 1. Đọc Ingested Artifacts (DVC Managed)
            train_path = os.path.join(self.ingest_dir, "train.csv")
            test_path = os.path.join(self.ingest_dir, "test.csv")
            
            logger.info("Đọc dữ liệu train và test từ Ingestion Layer...")
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            
            # 2. Pandera Schema Validation
            logger.info("Bắt đầu Pandera Schema Validation...")
            valid_train = run_validation_pipeline(
                df_train, mode="train", tracker=self.tracker, quarantine_dir=self.quarantine_dir
            )
            valid_test = run_validation_pipeline(
                df_test, mode="inference", tracker=self.tracker, quarantine_dir=self.quarantine_dir
            )
            
            # 3. Lưu Valid Artifacts cho DVC dựa vào đường dẫn trong yaml
            valid_train.to_csv(self.valid_train_path, index=False)
            valid_test.to_csv(self.valid_test_path, index=False)
            logger.info(f"Đã lưu valid artifacts tại {self.valid_train_path} và {self.valid_test_path}.")
            
            # 4. Evidently AI Drift Detection
            if os.path.exists(self.ref_data_path):
                logger.info("Bắt đầu Evidently AI Drift Detection...")
                df_ref = pd.read_csv(self.ref_data_path)
                
                drift_metrics, drift_report_path = run_drift_detection(
                    reference_data=df_ref, 
                    current_data=valid_train, 
                    output_dir=os.path.dirname(self.drift_report)
                )
                
                # MLflow Tracker: Log Drift Metrics
                self.tracker.log_params_and_metrics(metrics=drift_metrics)
                logger.info(f"Hoàn tất Drift Detection. Báo cáo lưu tại: {drift_report_path}")
            else:
                logger.warning(f"Không tìm thấy Reference Data tại {self.ref_data_path}. Bỏ qua Drift Detection.")
            
            logger.info("Toàn bộ Data Validation Pipeline hoàn tất thành công.")
            
        except Exception as e:
            self.tracker.log_error_and_tag(e)
            logger.error(f"Data Validation Pipeline thất bại: {e}")
            raise e
        finally:
            self.tracker.end_run()

if __name__ == "__main__":
    # Đọc tham số điều hướng từ file cấu hình YAML thay vì hardcode dict
    config_path = "config/pipelines/ml_calories/data_val.yaml"
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Không tìm thấy file cấu hình tại {config_path}")
        yaml_config = {}
        
    pipeline = CaloriesDataValidationPipeline(config_paths=yaml_config)
    pipeline.run()