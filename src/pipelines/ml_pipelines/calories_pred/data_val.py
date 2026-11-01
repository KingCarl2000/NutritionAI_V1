import os
import yaml
import importlib
import pandas as pd
import pandera as pa
from pandera.errors import SchemaErrors

from src.nutrition_core.logging.logger import logger
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker
from src.pipelines.ml_pipelines.calories_pred.eval_drift import CaloriesDriftEvaluator

# =============================================================================
# HELPER: DYNAMIC IMPORT TỪ API REGISTRY
# =============================================================================
def load_class_from_registry(api_path: str):
    """Load class động từ chuỗi import trong YAML để tránh hardcode import."""
    module_path, class_name = api_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

# =============================================================================
# MLOPS PIPELINE: PANDERA VALIDATION, QUARANTINE ROUTING & ALERTING
# =============================================================================
def run_validation_pipeline(
    df: pd.DataFrame, 
    schema_class, 
    mode: str, 
    tracker: MLOpsTracker, 
    error_threshold: float,
    quarantine_dir: str
) -> pd.DataFrame:
    """
    Thực thi Data Validation bằng Pandera với mọi tham số được truyền động từ config.
    """
    logger.info(f"Bắt đầu quá trình Data Validation (Chế độ: {mode.upper()})...")
    
    try:
        valid_df = schema_class.validate(df, lazy=True)
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
    def __init__(self, config_data: dict):
        # 1. Khởi tạo cấu hình
        self.api_registry = config_data["core_api_references"]
        self.paths = config_data["pipeline_paths"]
        self.params = config_data["validation_params"]

        # Load các class phụ thuộc từ API registry thay vì truy cập thuộc tính chưa khai báo.
        self.train_schema = load_class_from_registry(self.api_registry["train_schema"])
        self.infer_schema = load_class_from_registry(self.api_registry["infer_schema"])
        tracker = load_class_from_registry(self.api_registry["tracker"])
        self.tracker = tracker() if isinstance(tracker, type) else tracker
        
        # 2. Extract paths chuẩn xác từ YAML
        self.ingest_dir = self.paths["ingest_dir"]
        self.valid_train_path = self.paths["valid_train_data"]
        self.valid_test_path = self.paths["valid_test_data"]
        self.ref_data_path = self.paths["reference_data_path"]
        
        # 3. Quản lý thư mục báo cáo & quarantine tự động
        self.quarantine_dir = os.path.dirname(self.paths["validation_report"])
        self.drift_report = self.paths["drift_report"]
        self.error_threshold = self.params["error_threshold"]

    def run(self):
        self.tracker.start_run()
        try:
            train_path = os.path.join(self.ingest_dir, "train.csv")
            test_path = os.path.join(self.ingest_dir, "test.csv")
            
            logger.info("Đọc dữ liệu train và test từ Ingestion Layer...")
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            
            logger.info("Bắt đầu Pandera Schema Validation...")
            valid_train = run_validation_pipeline(
                df=df_train, 
                schema_class=self.train_schema,  # Truyền class schema đã load động
                mode="train", 
                tracker=self.tracker, 
                error_threshold=self.error_threshold,
                quarantine_dir=self.quarantine_dir
            )
            
            valid_test = run_validation_pipeline(
                df=df_test, 
                schema_class=self.infer_schema,  # Truyền class schema đã load động
                mode="inference", 
                tracker=self.tracker, 
                error_threshold=self.error_threshold,
                quarantine_dir=self.quarantine_dir
            )
            
            valid_train.to_csv(self.valid_train_path, index=False)
            valid_test.to_csv(self.valid_test_path, index=False)
            logger.info(f"Đã lưu valid artifacts tại {self.valid_train_path} và {self.valid_test_path}.")
            
            if os.path.exists(self.ref_data_path):
                logger.info("Bắt đầu Evidently AI Drift Detection...")
                df_ref = pd.read_csv(self.ref_data_path)
                
                drift_evaluator = CaloriesDriftEvaluator()
                evaluate_drift = getattr(drift_evaluator, "evaluate")
                drift_metrics, drift_report_path = evaluate_drift(
                    reference_data=df_ref,
                    current_data=valid_train,
                    output_dir=os.path.dirname(self.drift_report)
                )
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
    config_path = "config/pipelines/ml_calories/data_val.yaml"
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Không tìm thấy file cấu hình tại {config_path}")
        yaml_config = {}
        
    pipeline = CaloriesDataValidationPipeline(config_data=yaml_config)
    pipeline.run()