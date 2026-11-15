# Tài liệu cấu hình kiểm định dữ liệu — Calories Prediction

Tóm tắt các YAML cấu hình cho bước kiểm định dữ liệu và theo dõi drift của pipeline `ml_calories`.

### `data_val.yaml` (`config/pipelines/ml_calories/data_val.yaml`)
- **Vai trò:** Cấu hình schema Pandera, các thư mục dữ liệu của pipeline Calories và ngưỡng lỗi riêng cho bước kiểm định.
- **Cách gọi trong code:** `DataValidationConfig(pipeline_config_path="config/pipelines/ml_calories/data_val.yaml")` trong `src/pipelines/ml_pipelines/calories_pred/data_val.py`; lớp này nạp YAML và dùng `get_schema(mode="train" | "inference")` để tải schema theo chế độ chạy.
- **Cây cấu trúc (Schema rút gọn):**
  ```yaml
  schema_classes:
    train_schema: "config.pipelines.ml_calories.calories_schema.MlCaloriesTrainingSchema"
    inference_schema: "config.pipelines.ml_calories.calories_schema.MlCaloriesInputSchema"
  pipeline_paths:
    pandera_schema_path: "config/pipelines/ml_calories/calories_schema.py"
    evidently_ai_profile_path: "config/pipelines/ml_calories/evidently_profile.yaml"
    input_dir: "Artifacts/data/ml_calories/raw"
    output_dir: "Artifacts/data/ml_calories/validated"
    quarantine_dir: "Artifacts/data/ml_calories/quarantine"
    evidently_report_dir: "Artifacts/reports/ml_calories/evidently"
  validation_params:
    error_threshold: 0.05
  ```
- **Lưu ý:** Các đường dẫn schema là chuỗi định danh lớp dùng để nạp động. Trong mã `DataValidationConfig._extract_configs`, phần trích xuất thư mục được thể hiện là đọc `pipeline_paths.dvc_tracked_dirs.*`, trong khi YAML này đặt `input_dir`, `output_dir`, `quarantine_dir` trực tiếp dưới `pipeline_paths`; cần đồng bộ cấu trúc này để cấu hình đường dẫn được áp dụng đúng.

### `evidently_profile.yaml` (`config/pipelines/ml_calories/evidently_profile.yaml`)
- **Vai trò:** Cấu hình Evidently cho báo cáo và phát hiện drift dữ liệu của pipeline Calories.
- **Cách gọi trong code:** `CaloriesDriftEvaluator(evidently_config_path="config/pipelines/ml_calories/evidently_profile.yaml")` trong `src/pipelines/orchestrators.py`; lớp đánh giá drift được triển khai tại `src/pipelines/ml_pipelines/calories_pred/eval_drift.py`.
- **Cây cấu trúc (Schema rút gọn):**
  ```yaml
  engine_config:
    project_name: "Calories_Prediction_Drift_Monitoring"
    reference_data: "Artifacts/data/ml_calories/trainning/train.csv"
    report_dir: "Artifacts/reports/ml_calories/evidently"
    evidently_workspace_url: null
  column_mapping:
    target: "calories"
    id_column: "User_ID"
    prediction: "predicted_calories"
    datetime: "created_at"
    task: "regression"
    numerical_features: [age, weight, height, duration, heart_rate, body_temp]
    categorical_features: [gender]
  current_data:
    source: "postgresql"
    db_schema: "raw"
    table_name: "calories"
  data_drift:
    dataset_drift_share: 0.5
    num_stattest: "wasserstein"
    num_stattest_threshold: 0.1
    cat_stattest: "jensenshannon"
    cat_stattest_threshold: 0.1
    per_column_options: <object>
  ```
- **Lưu ý:** `per_column_options` cho phép cấu hình riêng phép kiểm định và ngưỡng, hiện có `age` (`ks`, `0.05`) và `gender` (`psi`, `0.1`). Đường dẫn tham chiếu hiện ghi `trainning`; xác nhận tên thư mục này khớp với dữ liệu thực tế.

### `global.yaml` (`config/pipelines/global.yaml`)
- **Vai trò:** Cấu hình dùng chung cho các pipeline, gồm đường dẫn API lõi, MLflow và các ngưỡng MLOps.
- **Cách gọi trong code:** `DataValidationConfig(global_config_path="config/pipelines/global.yaml")` trong `src/pipelines/ml_pipelines/calories_pred/data_val.py`; bước trích xuất cấu hình đọc `mlops_thresholds.data_failure_rate.info_threshold` và `warning_threshold`.
- **Cây cấu trúc (Schema rút gọn):**
  ```yaml
  core_api_references: <object>  # Đường dẫn chuỗi tới schema, checks, exceptions, decorators và tracker
  mlflow_config:
    tracking_uri: "sqlite:///mlflow.db"
    autologging: <object>
    artifacts: <object>
    requirements_mapping: <object>
  mlops_thresholds:
    data_failure_rate:
      info_threshold: 0.01
      warning_threshold: 0.05
    data_drift: <object>
    validation_sla: <object>
    service_availability: <object>
    api_error_rate: <object>
    service_latency: <object>
    resource_saturation: <object>
    data_quality_and_model: <object>
  ```
- **Lưu ý:** Ngưỡng tỷ lệ lỗi dữ liệu được dùng chung là 1% (info) và 5% (warning); `data_val.yaml` khai báo thêm `validation_params.error_threshold: 0.05` để cấu hình riêng cho pipeline.

## Module Python liên quan

### `logger.py` (`src/nutrition_core/logging/logger.py`)
- **Vai trò:** Khởi tạo logger dùng chung, ghi log theo mức độ và cung cấp decorator đo hiệu năng, xử lý lỗi DVC và xử lý lỗi kiểm định Pandera.
- **Import chuẩn:** `from src.nutrition_core.logging.logger import logger, PROJECT_ROOT, OPER_LOG_FILE_PATH, WARM_LOG_FILE_PATH, ERROR_LOG_FILE_PATH, monitor_performance, dvc_error_handler, validation_error_handler`
- **Public Interface (Stub):**
  ```python
  import logging
  from collections.abc import Callable
  from typing import Any

  PROJECT_ROOT: Path
  OPER_LOG_FILE_PATH: str
  WARM_LOG_FILE_PATH: str
  ERROR_LOG_FILE_PATH: str
  logger: logging.Logger

  def monitor_performance(time_limit: float = 5.0) -> Callable[..., Any]:
      """Decorator ghi thời gian chạy; cảnh báo nếu vượt ngưỡng."""
      ...

  def dvc_error_handler(
      max_retries: int = 3,
      retry_delay: int = 2,
      fallback_func: Callable[..., Any] | None = None,
  ) -> Callable[..., Any]:
      """Decorator ánh xạ lỗi DVC, thử lại lỗi remote và có thể chạy fallback."""
      ...

  def validation_error_handler() -> Callable[..., Any]:
      """Decorator ghi log lỗi Pandera/validation rồi ném lại ngoại lệ."""
      ...
  ```
- **Lưu ý:** Khi import module, nó tạo thư mục log theo ngày dưới `logs/oper_logs`, `logs/warm_logs`, `logs/error_logs` và gắn các file/console handlers cho logger. Decorator DVC tham chiếu các ngoại lệ trong `src.nutrition_core.exception.exception`; decorator validation bắt các lỗi Pandera và `DataValidationError`, sau đó vẫn raise lỗi để luồng gọi xử lý.

### `mlflow_tracker.py` (`src/nutrition_core/logging/mlflow_tracker.py`)
- **Vai trò:** Bao bọc các thao tác MLflow cho run, metrics, lineage, model, artifacts, system logs và báo cáo lỗi validation.
- **Import chuẩn:** `from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker`
- **Public Interface (Stub):**
  ```python
  from typing import Any
  import pandas as pd

  class MLOpsTracker:
      def __init__(self, experiment_name: str, tracking_uri: str = "sqlite:///mlflow.db") -> None: ...

      def start_run(self, run_name: str | None = None) -> Any:
          """Bắt đầu MLflow run và trả về đối tượng run/context manager."""
          ...

      def upload_system_logs(self) -> None: ...

      def log_params_and_metrics(
          self,
          params: dict[str, Any] | None = None,
          metrics: dict[str, float] | None = None,
          step: int | None = None,
      ) -> None: ...

      def log_error_and_tag(self, error_msg: str, tags: dict[str, str] | None = None) -> None: ...

      def log_dataset_lineage(self, df: pd.DataFrame, dataset_name: str, source_uri: str) -> None: ...

      def log_model(self, model: Any, artifact_path: str = "model", framework: str = "sklearn") -> None: ...

      def log_validation_errors(self, schema_name: str, errors: list[dict[str, Any]]) -> None: ...

      def end_run(self, status: str = "FINISHED") -> None: ...

      def log_artifact_file(self, local_path: str, artifact_path: str | None = None) -> None: ...
  ```
- **Lưu ý:** Gọi các phương thức ghi MLflow trong ngữ cảnh run đang hoạt động. Khởi tạo tracker thiết lập tracking URI/experiment và cố gắng bật system metrics; `end_run()` tải các file log lên MLflow trước khi kết thúc run. `log_validation_errors()` ghi JSON cục bộ dưới `Artifacts/reports/ml_calories/validation` rồi upload; đóng gói model đọc dependencies từ thư mục `requirements/`. `enable_llm_tracing` hiện nằm trong khối chuỗi nhiều dòng/comment nên không phải API khả dụng.
