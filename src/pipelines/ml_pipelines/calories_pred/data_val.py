import importlib
import json
import shutil
from pathlib import Path
from typing import Type, Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker
from src.nutrition_core.pandera.api.base import BaseSchema
from src.nutrition_core.pandera.backends.pandas_executor import NutritionPandasBackend
from src.nutrition_core.pandera.config import setup_pandera_validation
from src.nutrition_core.exception.val_exception import DataValidationError
from src.nutrition_core.logging.logger import logger
from src.pipelines.ml_pipelines.calories_pred.eval_drift import CaloriesDriftEvaluator


def load_class_from_registry(api_path: str) -> Any:
    """Load a class, function, or method from a dot-notation path in YAML."""
    if not api_path or not isinstance(api_path, str):
        raise ValueError("`api_path` phải là chuỗi ký tự hợp lệ.")

    parts = api_path.split(".")
    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        try:
            obj = importlib.import_module(module_path)
            for attr in parts[i:]:
                obj = getattr(obj, attr)
            return obj
        except ImportError:
            continue

    raise ImportError(f"Không thể import hoặc truy cập thành phần từ đường dẫn API: '{api_path}'")


class DataValidationConfig:
    """Load and expose configuration for the data-validation pipeline."""

    def __init__(
        self,
        global_config_path: str = "config/pipelines/global.yaml",
        pipeline_config_path: str = "config/pipelines/ml_calories/data_val.yaml",
        evidently_config_path: str = "config/pipelines/ml_calories/evidently_profile.yaml",
    ):
        self.base_dir = Path(__file__).resolve().parents[4]
        self.global_config = self._load_yaml(global_config_path)
        self.pipeline_config = self._load_yaml(pipeline_config_path)
        self.evidently_config = self._load_yaml(evidently_config_path)
        self._extract_configs()

    def _load_yaml(self, rel_path: str) -> Dict[str, Any]:
        """Read and parse a YAML configuration file."""
        file_path = self.base_dir / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def _extract_configs(self) -> None:
        """Extract pipeline paths, schema references, and shared thresholds."""
        pipeline_paths = self.pipeline_config.get("pipeline_paths", {})
        self.input_dir = self.base_dir / pipeline_paths["input_dir"]
        self.output_dir = self.base_dir / pipeline_paths["output_dir"]
        self.quarantine_dir = self.base_dir / pipeline_paths["quarantine_dir"]

        schema_classes = self.pipeline_config.get("schema_classes", {})
        self._train_schema_path = schema_classes.get("train_schema")
        self._inference_schema_path = schema_classes.get("inference_schema")

        failure_rates = self.global_config.get("mlops_thresholds", {}).get("data_failure_rate", {})
        self.info_threshold = failure_rates.get("info_threshold")
        self.warning_threshold = failure_rates.get("warning_threshold")

    def get_schema(self, mode: str = "train") -> Type[Any]:
        """Load the configured Pandera schema for training or inference."""
        if mode == "train":
            schema_path = self._train_schema_path
        elif mode == "inference":
            schema_path = self._inference_schema_path
        else:
            raise ValueError("Chế độ (mode) chỉ hỗ trợ 'train' hoặc 'inference'.")

        if not schema_path:
            raise ValueError(f"Không tìm thấy cấu hình schema cho mode '{mode}' trong data_val.yaml")
        return load_class_from_registry(schema_path)


def validate_with_pandas_backend(
    df: pd.DataFrame,
    schema_class: Type[Any],
    lazy: bool = True,
    tracker: Optional[MLOpsTracker] = None,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Use BaseSchema's shared implementation where supported; retain a backend fallback."""
    try:
        if isinstance(schema_class, type) and issubclass(schema_class, BaseSchema):
            validated_df, errors = schema_class.validate_df(
                df,
                lazy=lazy,
                drop_invalid_rows=False,
                mlflow_tracker=tracker,
                return_errors=True,
            )
            return validated_df, errors

        core_schema = schema_class.to_schema()
        backend = NutritionPandasBackend(
            schema=core_schema,
            lazy=lazy,
            drop_invalid_rows=False,
        )
        return backend.run_pipeline(df)
    except DataValidationError:
        raise
    except Exception as exc:
        logger.error(f"Lỗi không xác định trong quá trình validate backend: {exc}")
        raise DataValidationError(f"Backend Error: {exc}") from exc


def run_validation_pipeline(
    df: pd.DataFrame,
    schema_class: Type[Any],
    mode: str,
    tracker: MLOpsTracker,
    error_threshold: float,
    quarantine_dir: str,
    output_dir: str,
) -> pd.DataFrame:
    """Validate data, report row-level failures, and save the valid dataset."""
    import os

    logger.info(f"Bắt đầu Validation Pipeline (Mode: {mode}) với {len(df)} dòng dữ liệu.")
    validated_df, error_cases = validate_with_pandas_backend(
        df, schema_class, lazy=True, tracker=tracker
    )

    total_records = len(df)
    known_error_indices = set()
    unknown_index_error_count = 0

    for error in error_cases:
        error_index = error.get("index")
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

    invalid_record_count = (
        total_records if unknown_index_error_count > 0 else len(known_error_indices)
    )
    error_rate = invalid_record_count / total_records if total_records > 0 else 0.0

    report = {
        "mode": mode,
        "total_records": total_records,
        "error_count": invalid_record_count,
        "invalid_record_count": invalid_record_count,
        "unknown_index_error_count": unknown_index_error_count,
        "error_rate": error_rate,
        "errors_detail": error_cases[:100],
    }

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(quarantine_dir, exist_ok=True)

    report_path = os.path.join(output_dir, f"validation_report_{mode}.json")
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    if tracker:
        tracker.log_params_and_metrics(metrics={
            "validation_error_rate": error_rate,
            "validation_error_count": float(invalid_record_count),
        })
        tracker.log_artifact_file(report_path)

    if error_rate > error_threshold:
        quarantine_file = os.path.join(quarantine_dir, f"quarantine_{mode}.csv")
        df.to_csv(quarantine_file, index=False)
        logger.error(
            f"Validation thất bại! Tỷ lệ dòng lỗi {error_rate * 100:.2f}% "
            f"vượt ngưỡng {error_threshold * 100:.2f}%. Dữ liệu đã được đưa vào Quarantine."
        )
        if tracker:
            tracker.log_artifact_file(quarantine_file)
        raise DataValidationError(
            f"Tỷ lệ dòng vi phạm dữ liệu quá cao ({error_rate:.2%}). Dừng pipeline."
        )

    if invalid_record_count > 0:
        if unknown_index_error_count > 0:
            raise DataValidationError(
                "Không thể xác định index của một hoặc nhiều lỗi validation; "
                "không thể tạo tập dữ liệu sạch an toàn."
            )
        logger.warning(
            f"Validation hoàn thành với {invalid_record_count} dòng lỗi. "
            f"Tỷ lệ dòng lỗi ({error_rate * 100:.2f}%) nằm trong ngưỡng an toàn."
        )
        valid_df = df.drop(index=list(known_error_indices)).reset_index(drop=True)
    else:
        logger.info("Validation thành công tuyệt đối, không có dòng vi phạm.")
        valid_df = validated_df

    output_file = os.path.join(output_dir, f"{mode}.csv")
    valid_df.to_csv(output_file, index=False)
    logger.info(f"Đã lưu tập dữ liệu đạt chuẩn tại: {output_file}")
    return valid_df


if __name__ == "__main__":
    config_manager = DataValidationConfig()
    input_file = config_manager.input_dir / "train.csv"
    if not input_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu đầu vào để validation: {input_file}")

    current_df = pd.read_csv(input_file)

    train_schema = config_manager.get_schema(mode="train")
    validation_params = config_manager.pipeline_config.get("validation_params", {})
    error_threshold = validation_params.get("error_threshold")
    if error_threshold is None:
        error_threshold = config_manager.warning_threshold
    if error_threshold is None:
        raise ValueError(
            "Thiếu validation_params.error_threshold và "
            "mlops_thresholds.data_failure_rate.warning_threshold"
        )

    evidently_config = config_manager.evidently_config
    engine_config = evidently_config.get("engine_config", {})
    reference_data_value = engine_config.get("reference_data")
    if not reference_data_value:
        raise ValueError("Thiếu engine_config.reference_data trong evidently_profile.yaml")
    reference_file = config_manager.base_dir / reference_data_value
    if not reference_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu tham chiếu cho drift evaluation: {reference_file}")
    reference_df = pd.read_csv(reference_file)

    report_dir = config_manager.base_dir / "Artifacts/reports/ml_calories/validation"
    dvc_report_path = report_dir / "validation_report.json"
    report_dir.mkdir(parents=True, exist_ok=True)

    tracker = MLOpsTracker(experiment_name="calories_pred")
    try:
        with tracker.start_run(run_name="calories_validate_and_drift"):
            with setup_pandera_validation(mlflow_tracker=tracker):
                try:
                    validated_df = run_validation_pipeline(
                        df=current_df,
                        schema_class=train_schema,
                        mode="train",
                        tracker=tracker,
                        error_threshold=float(error_threshold),
                        quarantine_dir=str(config_manager.quarantine_dir),
                        output_dir=str(config_manager.output_dir),
                    )

                    drift_evaluator = CaloriesDriftEvaluator(
                        config=evidently_config,
                        mlops_tracker=tracker,
                    )
                    drift_evaluator.execute_evaluation(
                        current_df=validated_df,
                        reference_df=reference_df,
                    )
                finally:
                    generated_report = config_manager.output_dir / "validation_report_train.json"
                    if generated_report.is_file():
                        shutil.copy2(generated_report, dvc_report_path)
    except Exception:
        logger.exception("Calories validation and drift pipeline failed.")
        raise

    logger.info("Calories validation and drift evaluation completed successfully.")
