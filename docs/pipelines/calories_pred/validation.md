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
    current_data: "Artifacts/data/ml_calories/validated/train.csv"
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

### `exception.py` (`src/nutrition_core/exception/exception.py`)

- **Vai trò:** Định nghĩa các ngoại lệ dùng chung cho pipeline dữ liệu, ML và cơ sở dữ liệu; kèm thông tin file và dòng lỗi từ traceback.
- **Import chuẩn:**
  ```python
  from src.nutrition_core.exception.exception import (
      NutritionBaseException,
      DataPipelineException,
      DataTransformationError,
      DataCleansingError,
      MLPipelineException,
      ModelTrainingError,
      ModelEvaluationError,
      DatabaseException,
      DatabaseConnectionError,
      QueryExecutionError,
      BulkLoadError,
      CDCStreamError,
      DataNotFoundError,
      RemoteStorageError,
      VersionNotFoundError,
  )
  ```
- **Public Interface (Stub):**
  ```python
  import sys
  from types import ModuleType

  class NutritionBaseException(Exception):
      def __init__(
          self,
          error_message: str,
          error_details: ModuleType = sys,
      ) -> None: ...

      @staticmethod
      def _get_detailed_error_info(error_details: ModuleType): ...

      def __str__(self) -> str: ...

  class DataPipelineException(NutritionBaseException): ...
  class DataTransformationError(DataPipelineException): ...
  class DataCleansingError(DataPipelineException): ...

  class MLPipelineException(NutritionBaseException): ...
  class ModelTrainingError(MLPipelineException): ...
  class ModelEvaluationError(MLPipelineException): ...

  class DatabaseException(NutritionBaseException): ...
  class DatabaseConnectionError(DatabaseException): ...
  class QueryExecutionError(DatabaseException): ...
  class BulkLoadError(DatabaseException): ...
  class CDCStreamError(DatabaseException): ...

  class DataNotFoundError(DataPipelineException): ...
  class RemoteStorageError(DataPipelineException): ...
  class VersionNotFoundError(DataPipelineException): ...
  ```
- **Lưu ý:** `NutritionBaseException` lưu thông điệp lỗi trong `error_message` và cố gắng lấy tên file cùng số dòng từ `error_details.exc_info()`. Nếu không có traceback phù hợp, thông tin vị trí lỗi được đặt thành `"Unknown"`. Module cũng import `logger` từ `src.nutrition_core.logging`.

### `val_exception.py` (`src/nutrition_core/exception/val_exception.py`)

- **Vai trò:** Chuẩn hóa lỗi kiểm định Pandera thành `DataValidationError` và trích xuất chi tiết lỗi từ eager hoặc lazy validation.
- **Import chuẩn:**
  ```python
  from src.nutrition_core.exception.val_exception import (
      DataValidationError,
      PanderaErrorHandler,
  )
  ```
- **Public Interface (Stub):**
  ```python
  from typing import Any, Dict, List, Union
  from pandera.errors import SchemaError, SchemaErrors

  class DataValidationError(Exception):
      def __init__(
          self,
          message: str,
          errors: List[Dict[str, Any]] = None,
      ) -> None: ...
      # Thuộc tính: errors

  class PanderaErrorHandler:
      @staticmethod
      def parse_lazy_errors(
          err: SchemaErrors,
      ) -> List[Dict[str, Any]]: ...

      @staticmethod
      def parse_eager_error(
          err: SchemaError,
      ) -> List[Dict[str, Any]]: ...

      @classmethod
      def handle_validation_error(
          cls,
          err: Union[SchemaError, SchemaErrors],
          schema_name: str,
      ) -> DataValidationError: ...
  ```
- **Lưu ý:** `parse_lazy_errors` đọc các trường lỗi từ `failure_cases` dạng DataFrame; `parse_eager_error` trích xuất chi tiết lỗi đơn lẻ. `handle_validation_error` tạo và trả về `DataValidationError` cùng thông điệp có tên schema và danh sách lỗi đã chuẩn hóa. Module phụ thuộc vào Pandas và Pandera.

### `base.py` (`src/nutrition_core/pandera/api/base.py`)

- **Vai trò:** Cung cấp schema Pandera cơ sở và phương thức kiểm định DataFrame qua backend Pandas, đồng thời ghi kết quả và lỗi lên log/MLflow.
- **Import chuẩn:**
  ```python
  from src.nutrition_core.pandera.api.base import BaseSchema
  ```
- **Public Interface (Stub):**
  ```python
  from typing import Type, TypeVar
  import pandas as pd
  import pandera.pandas as pa

  SchemaModelT = TypeVar("SchemaModelT", bound=pa.DataFrameModel)

  class BaseSchema(pa.DataFrameModel):
      class Config:
          coerce = True
          strict = False
          ordered = False
          unique_column_names = True
          add_missing_columns = True

      @classmethod
      def validate_df(
          cls: Type[SchemaModelT],
          df: pd.DataFrame,
          lazy: bool = True,
          drop_invalid_rows: bool = False,
      ) -> pd.DataFrame: ...
  ```
- **Lưu ý:** `validate_df` chuyển schema thành `DataFrameSchema`, gọi `NutritionPandasBackend.run_pipeline` và ghi trạng thái, số lỗi cùng chi tiết validation lên MLflow. Khi phát hiện lỗi mà không bật `drop_invalid_rows`, phương thức ném `DataValidationError`.

### `check.py` (`src/nutrition_core/pandera/api/check.py`)

- **Vai trò:** Định nghĩa các custom check vectorized và factory tạo quy tắc kiểm định đa cột, theo nhóm, regex và cảnh báo outlier.
- **Import chuẩn:**
  ```python
  from src.nutrition_core.pandera.api.check import (
      is_in_numeric_range,
      is_valid_string_length,
      GenericCheckFactory,
  )
  ```
- **Public Interface (Stub):**
  ```python
  from typing import Any, List, Optional
  import pandas as pd
  import pandera.pandas as pa

  def is_in_numeric_range(
      pandas_obj: pd.Series,
      *,
      min_val: float = 0.0,
      max_val: float = 1e9,
  ) -> pd.Series: ...

  def is_valid_string_length(
      pandas_obj: pd.Series,
      *,
      min_len: int = 1,
      max_len: int = 255,
  ) -> pd.Series: ...

  class GenericCheckFactory:
      @staticmethod
      def check_sum_le_total(
          component_cols: List[str],
          total_col: str,
          error_msg: Optional[str] = None,
      ) -> pa.Check: ...

      @staticmethod
      def check_range_validity(
          min_col: str,
          target_col: str,
          max_col: str,
          error_msg: Optional[str] = None,
      ) -> pa.Check: ...

      @staticmethod
      def check_group_mean_comparison(
          groupby_col: str,
          target_col: str,
          lower_group: Any,
          higher_group: Any,
          error_msg: Optional[str] = None,
      ) -> pa.Check: ...

      @staticmethod
      def check_regex_pattern(
          pattern: str,
          error_msg: Optional[str] = None,
      ) -> pa.Check: ...

      @staticmethod
      def create_outlier_warning(
          upper_threshold: float,
          warning_msg: Optional[str] = None,
      ) -> pa.Check: ...
  ```
- **Lưu ý:** Hai hàm kiểm tra số và độ dài chuỗi được đăng ký làm Pandera check methods. Các factory trả về `pa.Check` để schema của pipeline có thể cấu hình và tái sử dụng.

### `pandas_executor.py` (`src/nutrition_core/pandera/backends/pandas_executor.py`)

- **Vai trò:** Thực thi kiểm định `pandas.DataFrame`, bao gồm kiểm tra đầu vào, ép kiểu, gom lỗi và tùy chọn loại bỏ các dòng không hợp lệ.
- **Import chuẩn:**
  ```python
  from src.nutrition_core.pandera.backends.pandas_executor import (
      NutritionPandasBackend,
      validate_with_pandas_backend,
  )
  ```
- **Public Interface (Stub):**
  ```python
  from typing import Any, Dict, List, Tuple
  import pandas as pd
  import pandera as pa

  class NutritionPandasBackend:
      def __init__(
          self,
          schema: pa.DataFrameSchema,
          lazy: bool = True,
          drop_invalid_rows: bool = False,
      ) -> None: ...

      @classmethod
      def resolve_backend(cls, data: Any) -> bool: ...

      def run_pipeline(
          self,
          df: pd.DataFrame,
      ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]: ...

  def validate_with_pandas_backend(
      schema: pa.DataFrameSchema,
      df: pd.DataFrame,
      **kwargs,
  ) -> pd.DataFrame: ...
  ```
- **Lưu ý:** Backend làm việc trên bản sao của DataFrame khi tiền xử lý và ép kiểu. `resolve_backend` ném `TypeError` nếu đầu vào không phải `pandas.DataFrame`. Khi `drop_invalid_rows=True`, các dòng bị đánh dấu không hợp lệ sẽ bị loại bỏ; nếu không, lỗi được trả về hoặc ném dưới dạng `DataValidationError` tùy cách gọi và chế độ validation.

### `config.py` (`src/nutrition_core/pandera/config.py`)

- **Vai trò:** Cấu hình mức độ kiểm định Pandera toàn cục dựa trên biến môi trường và tùy chọn ghi cấu hình lên MLflow.
- **Import chuẩn:**
  ```python
  from src.nutrition_core.pandera.config import setup_pandera_validation
  ```
- **Public Interface (Stub):**
  ```python
  from typing import Optional
  from pandera.config import config_context
  from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker

  def setup_pandera_validation(
      mlflow_tracker: Optional[MLOpsTracker] = None,
  ) -> config_context: ...
  ```
- **Lưu ý:** Đọc biến môi trường `PANDERA_VALIDATION_DEPTH`. Các giá trị hỗ trợ là `SCHEMA_ONLY`, `DATA_ONLY` và `SCHEMA_AND_DATA`; nếu giá trị thiếu hoặc không hợp lệ, mặc định dùng `SCHEMA_AND_DATA`. Hàm được bọc bởi `monitor_performance(time_limit=2.0)` và trả về `config_context` để áp dụng cấu hình.

### `decorators.py` (`src/nutrition_core/pandera/decorators.py`)

- **Vai trò:** Cung cấp decorator kiểm định kiểu dữ liệu hoặc I/O bằng Pandera, chuẩn hóa lỗi thành `DataValidationError` và tùy chọn ghi nhận lỗi validation.
- **Import chuẩn:**
  ```python
  from src.nutrition_core.pandera.decorators import (
      nutrition_check_types,
      nutrition_check_io,
  )
  ```
- **Public Interface (Stub):**
  ```python
  from typing import Any, Callable

  def nutrition_check_types(*args: Any, **kwargs: Any) -> Callable: ...

  def nutrition_check_io(
      head: int = 5,
      tail: int = 5,
      **kwargs: Any,
  ) -> Callable: ...
  ```
- **Lưu ý:** `nutrition_check_types` bọc `pandera.check_types`; `nutrition_check_io` bọc `pandera.check_io`. Hai decorator ghi log và chuyển `SchemaError`/`SchemaErrors` thành `DataValidationError`. Trong mã hiện tại, các nhánh xử lý lỗi gọi `PanderaErrorHandler.extract_errors`; hãy bảo đảm phương thức này được định nghĩa hoặc cập nhật cách gọi cho khớp với `val_exception.py`.

### `evidently_engine.py` (`src/nutrition_core/pandera/evidently_engine.py`)

- **Vai trò:** Kết hợp kiểm định schema Pandera với phát hiện drift, kiểm tra chất lượng dữ liệu, Evidently Workspace và ghi metrics lên MLflow.
- **Import chuẩn:**
  ```python
  from src.nutrition_core.pandera.evidently_engine import EvidentlyDriftEngine
  ```
- **Public Interface (Stub):**
  ```python
  from typing import Any, Dict, Optional, Tuple
  import pandas as pd
  import pandera as pa

  class EvidentlyDriftEngine:
      def __init__(
          self,
          artifact_dir: str,
          project_name: str,
          evidently_workspace_url: Optional[str] = None,
          mlops_tracker: Optional[Any] = None,
      ) -> None: ...

      def validate_schema(
          self,
          df: pd.DataFrame,
          schema_model: pa.DataFrameModel,
      ) -> Tuple[bool, Optional[pd.DataFrame]]: ...

      @staticmethod
      def _configure_workspace_dashboard(
          project: Any,
          column_mapping_cfg: Dict[str, Any],
      ) -> None: ...

      def run_drift_detection(
          self,
          current_df: pd.DataFrame,
          reference_df: pd.DataFrame,
          column_mapping_cfg: Dict[str, Any],
          drift_options_cfg: Dict[str, Any],
          **kwargs: Any,
      ) -> Dict[str, Any]: ...

      def log_metrics_to_mlflow(
          self,
          report_json: Dict[str, Any],
          test_suite_json: Optional[Dict[str, Any]] = None,
      ) -> None: ...
  ```
- **Lưu ý:** Khởi tạo engine tạo thư mục `artifact_dir` nếu chưa tồn tại. `run_drift_detection` nhận cấu hình bổ sung qua `kwargs` (ví dụ: chất lượng dữ liệu, đánh giá regression/classification, LLM, RecSys và test suite) và trả về report, snapshot cùng test suite nếu có. Đồng bộ Workspace chỉ được thực hiện khi có `evidently_workspace_url`; ghi metrics MLflow chỉ hoạt động khi tracker được truyền vào.

### `dvc_handler.py` (`src/data_io/dvc_handler.py`)

- **Vai trò:** Cung cấp adapter cho DVC để đồng bộ dữ liệu, tải dataset/mô hình/metrics/tệp ảnh và chuẩn hóa metrics bằng Pydantic.
- **Import chuẩn:**
  ```python
  from src.data_io.dvc_handler import (
      DVCHandler,
      DVCMetricsNormalizedSchema,
      GlobalSummaryMetrics,
      PerClassMetric,
      EpochMetrics,
      ExecutionMetadata,
  )
  ```
- **Public Interface (Stub):**
  ```python
  from typing import Any, Dict, Optional
  from pydantic import BaseModel

  class PerClassMetric(BaseModel):
      precision: float
      recall: float
      f1_score: float
      support: int

  class GlobalSummaryMetrics(BaseModel):
      accuracy: float
      f1_score: float
      precision: float
      recall: float
      loss: float
      roc_auc: Optional[float] = None

  class EpochMetrics(BaseModel):
      best_epoch: int
      total_epochs: int
      train_loss_final: float
      val_loss_final: float

  class ExecutionMetadata(BaseModel):
      dataset_version: Optional[str] = None
      execution_time_seconds: float
      num_samples_eval: int
      framework: Optional[str] = None

  class DVCMetricsNormalizedSchema(BaseModel):
      summary: GlobalSummaryMetrics
      class_metrics: Optional[Dict[str, PerClassMetric]] = None
      epoch_metrics: Optional[EpochMetrics] = None
      metadata: Optional[ExecutionMetadata] = None

  class DVCHandler:
      def __init__(
          self,
          repo: Optional[str] = None,
          rev: Optional[str] = None,
      ) -> None: ...

      def sync_data(self) -> None: ...
      def is_data_updated(self) -> bool: ...

      def get_dataset(
          self,
          data_path: str,
          format_type: str = "csv",
      ) -> pd.DataFrame: ...

      def get_latest_model(
          self,
          model_path: str,
          framework: str = "sklearn",
      ) -> Any: ...

      def get_metrics(
          self,
          metrics_path: str = "metrics.json",
      ) -> DVCMetricsNormalizedSchema: ...

      def get_image_base64(self, image_path: str) -> str: ...
      def add_data(self, file_path: str) -> None: ...
  ```
- **Lưu ý:** Các thao tác DVC được bọc bởi `dvc_error_handler` để xử lý thử lại và lỗi. `get_dataset` hỗ trợ CSV và Parquet; nếu tải thất bại, decorator được cấu hình với hàm fallback đọc dữ liệu từ `data/raw/`. `get_latest_model` hỗ trợ `sklearn` và `pytorch`. `add_data` hiện chỉ ghi cảnh báo; lệnh `dvc add` đã bị vô hiệu hóa trong thân hàm.

### `eval_drift.py` (`src/pipelines/ml_pipelines/calories_pred/eval_drift.py`)

- **Vai trò:** Điều phối đánh giá drift cho pipeline Calories, chuẩn hóa cấu hình, gọi `EvidentlyDriftEngine` và xuất báo cáo cùng metrics.
- **Import chuẩn:**
  ```python
  from src.pipelines.ml_pipelines.calories_pred.eval_drift import CaloriesDriftEvaluator
  ```
- **Public Interface (Stub):**
  ```python
  from typing import Any, Dict, Optional
  import pandas as pd
  from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker

  class CaloriesDriftEvaluator:
      def __init__(
          self,
          config: Dict[str, Any],
          mlops_tracker: Optional[MLOpsTracker] = None,
      ) -> None: ...

      def execute_evaluation(
          self,
          current_df: pd.DataFrame,
          reference_df: pd.DataFrame,
      ) -> Dict[str, Any]: ...
  ```
- **Lưu ý:** `config` là cấu hình đã được nạp (thường từ `evidently_profile.yaml`). Bộ đánh giá chuẩn hóa mapping cột và tùy chọn drift trước khi gọi engine. Khi có snapshot, xuất HTML, JSON, serialization đầy đủ và CSV vào thư mục báo cáo; nếu có test suite thì cũng xuất báo cáo HTML. Tracker MLflow là tùy chọn. Trong cấu hình `per_column_options`, tên phép kiểm định được chuyển tiếp, nhưng ngưỡng riêng từng cột hiện chưa được engine sử dụng.
