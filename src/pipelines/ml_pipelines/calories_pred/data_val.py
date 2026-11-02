import os
import yaml
import importlib
import pandas as pd
import pandera as pa
from pandera.errors import SchemaErrors

from src.nutrition_core.logging.logger import logger
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker

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
    quarantine_dir: str,
    output_dir: str
) -> pd.DataFrame:
    """
    Thực thi Data Validation bằng Pandera. Lọc dữ liệu lỗi vào quarantine_dir 
    và lưu dữ liệu sạch vào output_dir.
    """
    logger.info(f"Bắt đầu quá trình Data Validation (Chế độ: {mode.upper()})...")
    
    # Tạo sẵn các thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(quarantine_dir, exist_ok=True)
    
    valid_data_path = os.path.join(output_dir, f"validated_{mode}.parquet")

    try:
        # Lazy=True để Pandera quét toàn bộ lỗi thay vì dừng ở lỗi đầu tiên
        valid_df = schema_class.validate(df, lazy=True)
        logger.info(f"Validation thành công 100%. {len(valid_df)} bản ghi hợp lệ.")
        
        if tracker:
            tracker.log_params_and_metrics(metrics={f"val_{mode}_error_rate": 0.0})
            
        # LƯU FILE SẠCH RA ĐĨA CHO BƯỚC TRANSFORMATION
        valid_df.to_parquet(valid_data_path, index=False)
        logger.info(f"Đã lưu dữ liệu hợp lệ tại {valid_data_path}")
        
        return valid_df

    except SchemaErrors as err:
        # Trích xuất index của các bản ghi lỗi
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
            tracker.log_params_and_metrics(metrics={
                f"val_{mode}_error_rate": error_rate,
                f"val_{mode}_invalid_count": invalid_count
            })
            
        # Lưu dữ liệu lỗi (Quarantine) và Báo cáo lỗi chi tiết
        quarantine_data_path = os.path.join(quarantine_dir, f"quarantine_{mode}.csv")
        invalid_df.to_csv(quarantine_data_path, index=False)
        
        report_path = os.path.join(quarantine_dir, f"error_report_{mode}.csv")
        err.failure_cases.to_csv(report_path, index=False)
        
        logger.info(f"Đã cách ly dữ liệu lỗi tại {quarantine_data_path} và báo cáo tại {report_path}")
        
        # Vẫn lưu phần dữ liệu đúng (cứu vãn) nếu có bản ghi hợp lệ
        if not valid_df.empty:
            valid_df.to_parquet(valid_data_path, index=False)
            logger.info(f"Đã cứu vãn và lưu {len(valid_df)} dữ liệu hợp lệ tại {valid_data_path}")
        
        # Alert nếu lỗi vượt ngưỡng
        if error_rate > error_threshold:
            alert_msg = f"DATA ALERT: Tỷ lệ lỗi ({error_rate:.2%}) vượt ngưỡng cho phép ({error_threshold:.2%}). Dừng pipeline!"
            logger.error(alert_msg)
            raise Exception(alert_msg)
            
        return valid_df

# =============================================================================
# LỚP ĐIỀU PHỐI (ORCHESTRATOR CLASS) CHO BƯỚC VALIDATION
# =============================================================================
class CaloriesDataValidationPipeline:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        # Khớp chính xác với các key trong file data_val.yaml đã sửa
        self.api_refs = self.config.get("core_api_references", {})
        self.paths = self.config.get("pipeline_paths", {})
        self.params = self.config.get("validation_params", {})
        
        # Đường dẫn thư mục
        self.input_dir = self.paths.get("input_dir", "Artifacts/data/ml_calories/raw")
        self.output_dir = self.paths.get("output_dir", "Artifacts/data/ml_calories/validated")
        self.quarantine_dir = self.paths.get("quarantine_dir", "Artifacts/data/ml_calories/quarantine")
        
        # Tham số cấu hình
        self.error_threshold = self.params.get("error_threshold", 0.05)
        
        # Khởi tạo MLflow Tracker
        self.tracker = MLOpsTracker(experiment_name="calories_data_pipeline")
        
    def run(self):
        logger.info("Khởi chạy Calories Data Validation Pipeline...")
        # Bật tracking trên MLflow
        self.tracker.start_run(run_name="data_validation_stage")
        
        try:
            # Load class schema động
            schema_class_path = self.api_refs.get("train_schema", "config.pipelines.ml_calories.calories_schema.MlCaloriesTrainingSchema")
            schema_class = load_class_from_registry(schema_class_path)
            
            # Quét và validate tất cả file .parquet trong thư mục raw (train, val, test)
            if not os.path.exists(self.input_dir):
                raise FileNotFoundError(f"Không tìm thấy thư mục input: {self.input_dir}")
                
            file_processed = 0
            for file_name in os.listdir(self.input_dir):
                if file_name.endswith(".parquet"):
                    mode = file_name.replace(".parquet", "") # Lấy tên làm mode (train, test, val)
                    file_path = os.path.join(self.input_dir, file_name)
                    
                    logger.info(f"Tiến hành đọc dữ liệu từ: {file_path}")
                    df = pd.read_parquet(file_path)
                    
                    run_validation_pipeline(
                        df=df,
                        schema_class=schema_class,
                        mode=mode,
                        tracker=self.tracker,
                        error_threshold=self.error_threshold,
                        quarantine_dir=self.quarantine_dir,
                        output_dir=self.output_dir
                    )
                    file_processed += 1
            
            if file_processed == 0:
                logger.warning(f"Không tìm thấy file .parquet nào trong {self.input_dir} để validate!")
                
            self.tracker.end_run()
            logger.info("Hoàn tất Data Validation Pipeline.")
            
        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng trong quá trình validation: {str(e)}")
            self.tracker.end_run(status="FAILED")
            raise e

if __name__ == "__main__":
    config_file = "config/pipelines/ml_calories/data_val.yaml"
    pipeline = CaloriesDataValidationPipeline(config_file)
    pipeline.run()