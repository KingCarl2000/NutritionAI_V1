import os
import yaml
import json
import importlib
import pandas as pd

from src.nutrition_core.logging.logger import logger
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker
from src.nutrition_core.exception.val_exception import DataValidationError

# CÁC MODULE LÕI ĐƯỢC ĐỊNH NGHĨA TRONG SRC.MD
from src.nutrition_core.pandera.config import setup_pandera_validation
from src.nutrition_core.pandera.backends.pandas_executor import validate_with_pandas_backend
# (Sau này bạn có thể import thêm EvidentlyDriftEngine tại đây cho Giai đoạn 2)

# =============================================================================
# HELPER: DYNAMIC IMPORT TỪ API REGISTRY
# =============================================================================
def load_class_from_registry(api_path: str):
    module_path, class_name = api_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

# =============================================================================
# MLOPS PIPELINE: PANDERA CORE BACKEND & QUARANTINE ROUTING
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
    Thực thi Giai đoạn 1: Data Validation bằng NutritionPandasBackend.
    """
    logger.info(f"Bắt đầu quá trình Data Validation (Chế độ: {mode.upper()})...")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(quarantine_dir, exist_ok=True)
    
    valid_data_path = os.path.join(output_dir, f"valid_{mode}.parquet")

    try:
        # GỌI ENGINE LÕI THAY VÌ GỌI TRỰC TIẾP SCHEMA:
        # Backend sẽ lo liệu tiền xử lý (coerce), kiểm định, và trả về dữ liệu sạch + danh sách lỗi đã parse.
        valid_df, failure_cases = validate_with_pandas_backend(
            schema=schema_class,
            df=df,
            lazy=True,
            drop_invalid_rows=True
        )

        total_records = len(df)
        invalid_count = len(failure_cases) if failure_cases else 0
        error_rate = invalid_count / total_records if total_records > 0 else 0

        # KỊCH BẢN PASS (Sạch 100%)
        if invalid_count == 0:
            logger.info(f"Validation thành công 100%. {len(valid_df)} bản ghi hợp lệ.")
            if tracker:
                tracker.log_params_and_metrics(metrics={f"val_{mode}_error_rate": 0.0})
            valid_df.to_parquet(valid_data_path, index=False)
            return valid_df

        # KỊCH BẢN FAIL MỘT PHẦN (Có dữ liệu vi phạm)
        logger.warning(
            f"Phát hiện {invalid_count}/{total_records} bản ghi vi phạm logic "
            f"(Tỷ lệ lỗi: {error_rate:.2%})."
        )
        
        if tracker:
            tracker.log_params_and_metrics(metrics={
                f"val_{mode}_error_rate": error_rate,
                f"val_{mode}_invalid_count": invalid_count
            })
            
        # Trích xuất dữ liệu lỗi gốc dựa trên index của DataFrame sạch
        invalid_df = df.loc[~df.index.isin(valid_df.index)]
        
        # Lưu dữ liệu lỗi (Quarantine) và Báo cáo Validation (JSON) để DVC tracking
        quarantine_data_path = os.path.join(quarantine_dir, f"quarantine_{mode}.csv")
        invalid_df.to_csv(quarantine_data_path, index=False)
        
        report_path = os.path.join(quarantine_dir, f"validation_report_{mode}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(failure_cases, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Đã cách ly dữ liệu lỗi tại {quarantine_data_path} và báo cáo chi tiết tại {report_path}")
        
        # Cứu vãn dữ liệu hợp lệ còn lại
        if not valid_df.empty:
            valid_df.to_parquet(valid_data_path, index=False)
            logger.info(f"Đã cứu vãn và lưu {len(valid_df)} dữ liệu hợp lệ tại {valid_data_path}")
        
        # Ngắt Pipeline nếu lỗi vượt ngưỡng
        if error_rate > error_threshold:
            alert_msg = f"DATA ALERT: Tỷ lệ lỗi ({error_rate:.2%}) vượt ngưỡng cho phép ({error_threshold:.2%}). Dừng pipeline!"
            logger.error(alert_msg)
            raise Exception(alert_msg)
            
        return valid_df

    except DataValidationError as e:
        # Bắt ngoại lệ chuẩn hệ thống nếu lazy=False hoặc có lỗi tiền xử lý nghiêm trọng
        logger.error(f"[VALIDATION FATAL ERROR] {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Lỗi hệ thống không mong đợi trong Data Validation: {e}")
        raise e

# =============================================================================
# LỚP ĐIỀU PHỐI (ORCHESTRATOR CLASS)
# =============================================================================
class CaloriesDataValidationPipeline:
    def __init__(self, local_config_path: str, global_config_path: str):
        with open(global_config_path, "r", encoding="utf-8") as fg:
            self.global_config = yaml.safe_load(fg)
        with open(local_config_path, "r", encoding="utf-8") as fl:
            self.local_config = yaml.safe_load(fl)
            
        self.mlops_thresholds = self.global_config.get("mlops_thresholds", {})
        mlflow_config = self.global_config.get("mlflow_config", {})
        tracking_uri = mlflow_config.get("tracking_uri", "sqlite:///mlflow.db")
        
        self.schema_classes = self.local_config.get("schema_classes", {})
        self.paths = self.local_config.get("pipeline_paths", {})
        self.params = self.local_config.get("validation_params", {})
        
        self.input_dir = self.paths.get("input_dir")
        self.output_dir = self.paths.get("output_dir")
        self.quarantine_dir = self.paths.get("quarantine_dir")
        
        global_warning_threshold = self.mlops_thresholds.get("data_failure_rate", {}).get("warning_threshold", 0.05)
        self.error_threshold = self.params.get("error_threshold", global_warning_threshold)
        
        self.tracker = MLOpsTracker(
            experiment_name="calories_data_pipeline", 
            tracking_uri=tracking_uri
        )
        
    def run(self):
        logger.info("Khởi chạy Calories Data Validation Pipeline...")
        self.tracker.start_run(run_name="data_validation_stage")
        
        try:
            # 1. KHỞI TẠO CẤU HÌNH TOÀN CỤC CHO PANDERA (Từ config.py)
            setup_pandera_validation(mlflow_tracker=self.tracker)
            
            if not self.input_dir or not os.path.exists(self.input_dir):
                raise FileNotFoundError(f"Thư mục input không tồn tại: {self.input_dir}")
                
            schema_class_path = self.schema_classes.get("train_schema", "config.pipelines.ml_calories.calories_schema.MlCaloriesTrainingSchema")
            schema_class = load_class_from_registry(schema_class_path)
                
            file_processed = 0
            for file_name in os.listdir(self.input_dir):
                if file_name.endswith(".parquet"):
                    mode = file_name.replace("raw_", "").replace(".parquet", "")
                    file_path = os.path.join(self.input_dir, file_name)
                    
                    logger.info(f"Tiến hành đọc dữ liệu từ: {file_path}")
                    df = pd.read_parquet(file_path)
                    
                    # Giai đoạn 1: Pandera Validation
                    valid_df = run_validation_pipeline(
                        df=df,
                        schema_class=schema_class,
                        mode=mode,
                        tracker=self.tracker,
                        error_threshold=self.error_threshold,
                        quarantine_dir=self.quarantine_dir,
                        output_dir=self.output_dir
                    )
                    
                    # Giai đoạn 2: [PLACEHOLDER CHO EVIDENTLY AI DRIFT DETECTION]
                    # Nếu chạy file test/val, bạn có thể truyền tiếp valid_df vào EvidentlyDriftEngine 
                    # để xuất drift_report.html như minh họa trong kiến trúc DVC.
                    
                    file_processed += 1
            
            if file_processed == 0:
                logger.warning(f"Không tìm thấy file .parquet nào trong {self.input_dir}!")
                
            self.tracker.end_run()
            logger.info("Hoàn tất Data Validation Pipeline.")
            
        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng trong quá trình validation: {str(e)}")
            self.tracker.end_run(status="FAILED")
            raise e

if __name__ == "__main__":
    LOCAL_CONFIG_PATH = "config/pipelines/ml_calories/data_val.yaml"
    GLOBAL_CONFIG_PATH = "config/pipelines/global.yaml"
    
    pipeline = CaloriesDataValidationPipeline(
        local_config_path=LOCAL_CONFIG_PATH, 
        global_config_path=GLOBAL_CONFIG_PATH
    )
    pipeline.run()