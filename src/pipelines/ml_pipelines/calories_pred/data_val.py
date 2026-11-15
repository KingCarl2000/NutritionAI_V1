
import importlib
import yaml
from pathlib import Path
from typing import Type, Any, Dict, List, Tuple


import json
import pandas as pd

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
        # 1. Đường dẫn thư mục DVC được khai báo trực tiếp dưới pipeline_paths.
        pipeline_paths = self.pipeline_config.get("pipeline_paths", {})
        self.input_dir = self.base_dir / pipeline_paths.get("input_dir")
        self.output_dir = self.base_dir / pipeline_paths.get("output_dir")
        self.quarantine_dir = self.base_dir / pipeline_paths.get("quarantine_dir")
        
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
    Validate DataFrame bằng NutritionPandasBackend, bao gồm tiền xử lý,
    ép kiểu, thực thi schema và thu thập các lỗi validation.
    """
    core_schema = schema_class.to_schema()
    backend = NutritionPandasBackend(
        schema=core_schema,
        lazy=lazy,
        drop_invalid_rows=False,
    )

    try:
        # Chạy toàn bộ pipeline của backend thay vì gọi trực tiếp core_schema.validate.
        validated_df, errors = backend.run_pipeline(df)
        return validated_df, errors
    except Exception as e:
        logger.error(f"Lỗi không xác định trong quá trình validate backend: {str(e)}")
        raise DataValidationError(f"Backend Error: {str(e)}") from e


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
    Tỷ lệ lỗi tính theo số dòng lỗi duy nhất; lỗi không gắn được với một index cụ thể
    được xem là lỗi toàn tập dữ liệu để tránh đánh giá thiếu mức độ nghiêm trọng.
    """
    import os
    
    logger.info(f"Bắt đầu Validation Pipeline (Mode: {mode}) với {len(df)} dòng dữ liệu.")
    
    # 1. Thực thi kiểm định qua Pandas Backend
    validated_df, error_cases = validate_with_pandas_backend(df, schema_class, lazy=True)
    
    total_records = len(df)
    known_error_indices = set()
    unknown_index_error_count = 0

    for error in error_cases:
        error_index = error.get("index")
        # Các lỗi thiếu index, null, hoặc index không có trong DataFrame
        # không thể ánh xạ an toàn tới một dòng cụ thể.
        try:
            index_is_valid = (
                error_index is not None
                and not pd.isna(error_index)
                and error_index in df.index
            )
        except (TypeError, ValueError):
            index_is_valid = False

        if index_is_valid:
            known_error_indices.add(error_index)
        else:
            unknown_index_error_count += 1

    # Nếu có lỗi không thể gắn với dòng cụ thể (ví dụ thiếu cột bắt buộc),
    # xem toàn bộ tập dữ liệu là không hợp lệ. Đây là cách bảo thủ để không
    # cho phép pipeline tiếp tục chỉ vì lỗi schema thiếu thông tin index.
    invalid_record_count = (
        total_records
        if unknown_index_error_count > 0
        else len(known_error_indices)
    )
    error_rate = invalid_record_count / total_records if total_records > 0 else 0.0
    
    # 2. Tạo báo cáo Validation Report
    report = {
        "mode": mode,
        "total_records": total_records,
        "error_count": invalid_record_count,
        "invalid_record_count": invalid_record_count,
        "unknown_index_error_count": unknown_index_error_count,
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
        tracker.log_params_and_metrics(metrics={
            "validation_error_rate": error_rate,
            "validation_error_count": float(invalid_record_count),
        })
        tracker.log_artifact_file(report_path)

    # 3. Phân luồng dựa trên tỷ lệ dòng lỗi
    if error_rate > error_threshold:
        quarantine_file = os.path.join(quarantine_dir, f"quarantine_{mode}.csv")
        df.to_csv(quarantine_file, index=False)
        logger.error(
            f"Validation thất bại! Tỷ lệ dòng lỗi {error_rate*100:.2f}% "
            f"vượt ngưỡng {error_threshold*100:.2f}%. Dữ liệu đã được đưa vào Quarantine."
        )
        
        if tracker:
            tracker.log_artifact_file(quarantine_file)
            
        raise DataValidationError(f"Tỷ lệ dòng vi phạm dữ liệu quá cao ({error_rate:.2%}). Dừng pipeline.")
    
    elif invalid_record_count > 0:
        logger.warning(
            f"Validation hoàn thành với {invalid_record_count} dòng lỗi. "
            f"Tỷ lệ dòng lỗi ({error_rate*100:.2f}%) nằm trong ngưỡng an toàn."
        )
        if unknown_index_error_count > 0:
            # Không thể xác định các dòng hợp lệ một cách an toàn.
            raise DataValidationError(
                "Không thể xác định index của một hoặc nhiều lỗi validation; "
                "không thể tạo tập dữ liệu sạch an toàn."
            )
        valid_df = df.drop(index=list(known_error_indices)).reset_index(drop=True)
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
