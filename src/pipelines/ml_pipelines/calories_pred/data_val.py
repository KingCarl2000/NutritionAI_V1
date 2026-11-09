
import importlib
import yaml
from pathlib import Path
from typing import Type, Any, Dict


import json
import pandas as pd
import pandera.errors as pa_errors

# Giả định import các thành phần Core đã có trong dự án
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker
from src.nutrition_core.pandera.backends.pandas_executor import NutritionPandasBackend
from src.nutrition_core.exception.val_exception import DataValidationError
from src.nutrition_core.logging.logger import logger

def load_class_from_registry(api_path: str) -> Any:
    """
    Nạp động Class, Function hoặc Method từ đường dẫn dot-notation trong YAML.
    Hỗ trợ cả 2 dạng:
    - 'package.module.ClassName'
    - 'package.module.ClassName.method_name'
    """
    if not api_path or not isinstance(api_path, str):
        raise ValueError("`api_path` phải là chuỗi ký tự hợp lệ.")

    parts = api_path.split(".")
    
    # Thử import module từ dài nhất đến ngắn nhất
    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        try:
            obj = importlib.import_module(module_path)
            # Duyệt qua các attr còn lại (Class -> Method/Property)
            for attr in parts[i:]:
                obj = getattr(obj, attr)
            return obj
        except ImportError:
            continue
            
    raise ImportError(f"Không thể import hoặc truy cập thành phần từ đường dẫn API: '{api_path}'")


class DataValidationConfig:
    """
    Lớp quản lý và khởi tạo cấu hình cho quy trình Data Validation từ các file YAML.
    """
    def __init__(
        self, 
        global_config_path: str = "config/pipelines/global.yaml",
        pipeline_config_path: str = "config/pipelines/ml_calories/data_val.yaml",
        evidently_config_path: str = "config/pipelines/ml_calories/evidently_profile.yaml"
    ):
        self.base_dir = Path(__file__).resolve().parents[4] # Điều chỉnh tùy thuộc vào vị trí chạy lệnh
        
        # Load cấu hình từ các file YAML
        self.global_config = self._load_yaml(global_config_path)
        self.pipeline_config = self._load_yaml(pipeline_config_path)
        self.evidently_config = self._load_yaml(evidently_config_path)

        # Trích xuất các tham số quan trọng
        self._extract_configs()

    def _load_yaml(self, rel_path: str) -> Dict[str, Any]:
        """Đọc và parse file YAML."""
        file_path = self.base_dir / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _extract_configs(self):
        """Khởi tạo các tham số định tuyến và pipeline specific."""
        # 1. Đường dẫn thư mục DVC
        dvc_paths = self.pipeline_config.get("pipeline_paths", {}).get("dvc_tracked_dirs", {})
        self.input_dir = self.base_dir / dvc_paths.get("input_dir")
        self.output_dir = self.base_dir / dvc_paths.get("output_dir")
        self.quarantine_dir = self.base_dir / dvc_paths.get("quarantine_dir")
        
        # 2. Lấy API paths của Pandera Schema
        schema_classes = self.pipeline_config.get("schema_classes", {})
        self._train_schema_path = schema_classes.get("train_schema")
        self._inference_schema_path = schema_classes.get("inference_schema")
        
        # 3. MLOps Thresholds (từ global.yaml)
        failure_rates = self.global_config.get("mlops_thresholds", {}).get("data_failure_rate", {})
        self.info_threshold = failure_rates.get("info_threshold")
        self.warning_threshold = failure_rates.get("warning_threshold")

    def get_schema(self, mode: str = "train") -> Type[Any]:
        """
        Tải và trả về lớp Pandera Schema dựa trên chế độ thực thi.
        
        Args:
            mode (str): 'train' cho huấn luyện, 'inference' cho dự đoán.
        """
        if mode == "train":
            if not self._train_schema_path:
                raise ValueError("Không tìm thấy cấu hình 'train_schema' trong data_val.yaml")
            return load_class_from_registry(self._train_schema_path)
        elif mode == "inference":
            if not self._inference_schema_path:
                raise ValueError("Không tìm thấy cấu hình 'inference_schema' trong data_val.yaml")
            return load_class_from_registry(self._inference_schema_path)
        else:
            raise ValueError("Chế độ (mode) chỉ hỗ trợ 'train' hoặc 'inference'.")


def validate_with_pandas_backend(
    df: pd.DataFrame, 
    schema_class: Type[Any], 
    lazy: bool = True
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Hàm xử lý backend kiểm định thực tế cho Pandas.
    Sử dụng cơ chế Lazy Validation để gom toàn bộ lỗi thay vì dừng lại ở lỗi đầu tiên.
    
    Args:
        df (pd.DataFrame): Dữ liệu cần kiểm định.
        schema_class (Type[Any]): Lớp Pandera Schema (thường kế thừa BaseSchema).
        lazy (bool): Bật chế độ tổng hợp lỗi (Lazy Validation).
        
    Returns:
        Tuple[pd.DataFrame, List[dict]]: DataFrame sau khi xử lý (hoặc gốc) và danh sách chi tiết lỗi.
    """
    errors = []
    validated_df = df.copy()
    
    # Chuyển đổi DataFrameModel sang DataFrameSchema để tương thích với Backend (nếu cần)
    core_schema = schema_class.to_schema()
    
    # Khởi tạo backend thực thi
    backend = NutritionPandasBackend(schema=core_schema, lazy=lazy, drop_invalid_rows=False)
    
    try:
        # Tùy thuộc vào thiết kế nội bộ của NutritionPandasBackend, hàm thực thi có thể là validate hoặc _preprocess_and_coerce
        # Nếu Backend ném ra SchemaErrors (do lazy=True), ta sẽ bắt và trích xuất
        validated_df = core_schema.validate(df, lazy=lazy)
    except pa_errors.SchemaErrors as err:
        # Gom các lỗi định dạng thành danh sách dict dễ đọc cho validation_report
        for error in err.failure_cases.itertuples(index=False):
            errors.append({
                "column": getattr(error, "column", "N/A"),
                "check": getattr(error, "check", "N/A"),
                "failure_case": str(getattr(error, "failure_case", "N/A")),
                "index": getattr(error, "index", "N/A")
            })
        
        # Đồng bộ hóa với failure_cases của backend (tùy chọn)
        backend.failure_cases = errors
        
    except Exception as e:
        logger.error(f"Lỗi không xác định trong quá trình validate backend: {str(e)}")
        raise DataValidationError(f"Backend Error: {str(e)}")
        
    return validated_df, errors


def run_validation_pipeline(
    df: pd.DataFrame, 
    schema_class: Type[Any], 
    mode: str, 
    tracker: MLOpsTracker, 
    error_threshold: float, 
    quarantine_dir: str, 
    output_dir: str
) -> pd.DataFrame:
    """
    Hàm điều phối chính thực thi kiểm định dữ liệu.
    Xử lý tỷ lệ lỗi, ghi nhận vào MLflow Tracker, xuất file báo cáo và xử lý logic Fail-Fast/Quarantine.
    """
    import os
    
    logger.info(f"Bắt đầu Validation Pipeline (Mode: {mode}) với {len(df)} dòng dữ liệu.")
    
    # 1. Thực thi kiểm định qua Pandas Backend
    validated_df, error_cases = validate_with_pandas_backend(df, schema_class, lazy=True)
    
    total_records = len(df)
    error_count = len(error_cases)
    error_rate = error_count / total_records if total_records > 0 else 0.0
    
    # 2. Tạo báo cáo Validation Report
    report = {
        "mode": mode,
        "total_records": total_records,
        "error_count": error_count,
        "error_rate": error_rate,
        "errors_detail": error_cases[:100]  # Giới hạn ghi 100 lỗi đầu tiên để tránh file quá lớn
    }
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(quarantine_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, f"validation_report_{mode}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
        
    # Track metadata lên MLflow
    if tracker:
        tracker.log_metrics({"validation_error_rate": error_rate, "validation_error_count": error_count})
        tracker.log_artifact(report_path)

    # 3. Phân luồng dựa trên ngưỡng lỗi (Threshold Routing)
    if error_rate > error_threshold:
        # Fail-Fast hoặc chuyển vùng cách ly (Quarantine)
        quarantine_file = os.path.join(quarantine_dir, f"quarantine_{mode}.csv")
        df.to_csv(quarantine_file, index=False)
        logger.error(f"Validation thất bại! Tỷ lệ lỗi {error_rate*100:.2f}% vượt ngưỡng {error_threshold*100:.2f}%. Dữ liệu đã được đưa vào Quarantine.")
        
        # Track dataset hỏng nếu cần
        if tracker:
            tracker.log_artifact(quarantine_file)
            
        raise DataValidationError(f"Tỷ lệ vi phạm dữ liệu quá cao ({error_rate:.2%}). Dừng pipeline.")
    
    elif error_count > 0:
        logger.warning(f"Validation hoàn thành với {error_count} cảnh báo nhỏ. Tỷ lệ lỗi ({error_rate*100:.2f}%) nằm trong ngưỡng an toàn.")
        # Loại bỏ các dòng lỗi trước khi đưa ra bản sạch (logic tùy thuộc nghiệp vụ)
        if error_cases:
            error_indices = [err["index"] for err in error_cases if err.get("index") != "N/A"]
            valid_df = df.drop(index=error_indices).reset_index(drop=True)
        else:
            valid_df = validated_df
    else:
        logger.info("Validation thành công tuyệt đối, không có dòng vi phạm.")
        valid_df = validated_df

    # 4. Xuất file dữ liệu sạch
    output_file = os.path.join(output_dir, f"valid_{mode}.csv")
    valid_df.to_csv(output_file, index=False)
    logger.info(f"Đã lưu tập dữ liệu đạt chuẩn tại: {output_file}")
    
    return valid_df


# Test nhanh (Tùy chọn - Dùng để kiểm tra khi chạy trực tiếp file)
if __name__ == "__main__":
    try:
        config_manager = DataValidationConfig()
        
        # Nạp schema huấn luyện
        TrainSchema = config_manager.get_schema(mode="train")
        print(f"Đã nạp thành công Schema Huấn luyện: {TrainSchema.__name__}")
        
        print(f"Thư mục đầu ra (Validated): {config_manager.output_dir}")
        print(f"Ngưỡng Fail-Fast (Global): {config_manager.warning_threshold * 100}%")
        
    except Exception as e:
        print(f"Lỗi khởi tạo cấu hình: {e}")