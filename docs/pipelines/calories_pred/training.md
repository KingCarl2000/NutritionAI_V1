# Calories Prediction — Training Pipeline

## Tổng quan DVC pipeline

`src/scripts/setup_dvc_stages.sh` đăng ký các stage của quy trình Calories Prediction:

| Stage | Entrypoint | Input/output chính | Metrics |
| --- | --- | --- | --- |
| `calories_ingest` | `src/pipelines/ml_pipelines/calories_pred/data_ingest.py` | Tạo dữ liệu raw | `Artifacts/reports/ml_calories/ingestion/ingestion_report.json` |
| `calories_validate` | `src/pipelines/ml_pipelines/calories_pred/data_val.py` | Raw data → validated data | `Artifacts/reports/ml_calories/validation/validation_report.json` |
| `calories_transform` | `src/pipelines/ml_pipelines/calories_pred/data_trans.py` | Validated splits → `.npy` và preprocessor | `Artifacts/reports/ml_calories/transformation/transformation_report.json` |
| `calories_train` | `src/pipelines/ml_pipelines/calories_pred/model_train.py` | Train/validation `.npy` → model | `Artifacts/reports/ml_calories/training/training_report.json` |

Khởi tạo/cập nhật cấu hình các stage bằng `bash src/scripts/setup_dvc_stages.sh`, sau đó chạy riêng stage huấn luyện bằng `dvc repro calories_train`. Script dùng `dvc stage add --force` và tự chuyển về thư mục gốc repo.

## Stage `calories_train`

- **Dependencies DVC:** `model_train.py`, `model_factory.py`, `model_trainer.py`, `dvc_handler.py`, `logger.py`, `exception.py`; các input artifacts gồm `train.npy`, `val.npy`, `preprocessing.pkl` và transformation report.
- **Cấu hình DVC:** `config/pipelines/ml_calories/hydra_config.yaml` và `config/pipelines/ml_calories/train_params.yaml`.
- **Output:** `Artifacts/data/ml_calories/models`.
- **Metrics:** `Artifacts/reports/ml_calories/training/training_report.json`.
- **Luồng chạy:** Đọc train/validation arrays (cột cuối là target), kiểm tra dữ liệu hữu hạn và feature count so với preprocessor, chạy Optuna HPO cho từng metric được chọn, lưu model tốt nhất và JSON report. Validation split được dùng để chọn model; test split không được đọc trong stage này. Với cấu hình mặc định có nhiều metric, mỗi model được lưu thành `best_model_<metric_name>.pkl`; tên `best_model.pkl` chỉ được dùng nguyên trạng khi chạy một metric.
- **Chạy trực tiếp:** `python src/pipelines/ml_pipelines/calories_pred/model_train.py`

### Cấu hình YAML

#### `hydra_config.yaml` (`config/pipelines/ml_calories/hydra_config.yaml`)

- **Vai trò:** Khai báo cấu hình Hydra cho entrypoint huấn luyện, defaults, experiment settings, thư mục chạy, logging và cấu hình Optuna sweeper.
- **Cách gọi trong code:** `@hydra.main(config_path="../../../../config/pipelines/ml_calories", config_name="hydra_config")`; entrypoint nhận `cfg: DictConfig`.
- **Cây cấu trúc (Schema rút gọn):**
  ```yaml
  defaults: [_self_, train_params, override hydra/sweeper: optuna, override hydra/launcher: joblib]
  task:
    pipeline_name: calories_prediction
    debug: false
    seed: 42
  training:
    experiment_name: ${train_params.mlflow.experiment_name}
    tracking_uri: ${train_params.mlflow.tracking_uri}
    direction: auto
    metric_name: ${train_params.metric_name}
    metric_names: ${train_params.metric_names}
    n_trials: 20
    n_jobs: 1
    model_name: ${train_params.model_type}
    seed: ${task.seed}
  hydra:
    mode: RUN
    run.dir: Artifacts/runs/ml_calories/${now:%Y-%m-%d}/${now:%H-%M-%S}
    sweep.dir: Artifacts/runs/ml_calories/multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    sweeper:
      storage: sqlite:///Artifacts/optuna/calories.db
      study_name: calories_regression
      custom_search_space: src.pipelines.ml_pipelines.calories_pred.search_space.get_search_space
    launcher:
      n_jobs: 1
  ```
- **Lưu ý:** `hydra.mode` mặc định là `RUN`; cấu hình có sweeper/launcher cho Hydra multirun nhưng `model_train.py` gọi `ModelTrainer` để thực hiện Optuna HPO trong một lần chạy thường. `hydra.sweeper.params` có giá trị `ada_boosst`, không nằm trong các model alias của `ModelFactory`; không chọn giá trị này nếu bật cấu hình multirun.

#### `train_params.yaml` (`config/pipelines/ml_calories/train_params.yaml`)

- **Vai trò:** Cung cấp tham số huấn luyện, đường dẫn artifacts và thông tin tracking MLflow cho Calories Prediction.
- **Cách gọi trong code:** Được merge qua `defaults` của Hydra; truy cập bằng `cfg.train_params` hoặc các interpolation `${train_params...}` trong `hydra_config.yaml`.
- **Cây cấu trúc (Schema rút gọn):**
  ```yaml
  train_params:
    random_state: 42
    batch_size: 32
    learning_rate: 0.001
    n_estimators: 200
    max_depth: 6
    model_type: auto
    metric_name: auto
    metric_names: [mean_squared_error, r2_score, root_mean_squared_error]
    direction: auto
    artifacts:
      transformed_train_path: Artifacts/data/ml_calories/transformed/train.npy
      transformed_val_path: Artifacts/data/ml_calories/transformed/val.npy
      preprocessor_path: Artifacts/data/ml_calories/transformed/preprocessing.pkl
      training_output_dir: Artifacts/data/ml_calories/models
      best_model_path: Artifacts/data/ml_calories/models/best_model.pkl
      training_report_path: Artifacts/reports/ml_calories/training/training_report.json
    mlflow:
      experiment_name: calories_pred
      run_name: calories_training
      tracking_uri: ${oc.env:MLFLOW_TRACKING_URI,sqlite:///mlflow.db}
      log_params: true
      log_metrics: true
      register_model: true
  ```
- **Lưu ý:** `MLFLOW_TRACKING_URI` tùy chọn; khi không đặt, URI mặc định là `sqlite:///mlflow.db`. Trong `model_train.py`, các trường artifact, `model_type`, metric và experiment/tracking URI được dùng trực tiếp; một số giá trị như `random_state`, `batch_size`, `learning_rate`, `n_estimators`, `max_depth`, `log_params`, `log_metrics` và `register_model` không được entrypoint hiện tại đọc trực tiếp.

### Module Python

#### `model_train.py` (`src/pipelines/ml_pipelines/calories_pred/model_train.py`)

- **Vai trò:** Hydra entrypoint điều phối HPO cho một hoặc nhiều metric, kiểm tra input artifacts và ghi model/report của stage.
- **Import chuẩn:** `from src.pipelines.ml_pipelines.calories_pred.model_train import run_training`
- **Public Interface (Stub):**
  ```python
  PROJECT_ROOT: Path
  CONFIG_PATH: str
  METRICS: dict[str, tuple[Callable[[Any, Any], float], str]]

  def run_training(cfg: DictConfig) -> dict[str, Any]:
      """Huấn luyện/tối ưu model và ghi training report."""
      ...

  def main(cfg: DictConfig) -> None:
      """Hydra entrypoint gọi run_training(cfg)."""
      ...
  ```
- **Lưu ý:** Mỗi `.npy` phải là mảng 2D hữu hạn có ít nhất hai cột; cột cuối là target. Số feature phải khớp với `preprocessing.pkl`. Metric hỗ trợ gồm MSE, RMSE, RMSLE (minimize) và R² (maximize); `direction: auto` lấy hướng tối ưu từ metric.

#### `model_trainer.py` (`utils/ml_utils/model/model_trainer.py`)

- **Vai trò:** Chạy Optuna trials, quản lý nested MLflow runs, chọn trial tốt nhất và xuất model/params.
- **Import chuẩn:** `from utils.ml_utils.model.model_trainer import ModelTrainer`
- **Public Interface (Stub):**
  ```python
  class ModelTrainer:
      def __init__(
          self,
          experiment_name: str,
          direction: str = "maximize",
          eval_metric_fn: Callable[[Any, Any], float] | None = None,
          pruner: optuna.pruners.BasePruner | None = None,
          tracking_uri: str | None = "sqlite:///mlflow.db",
          seed: int = 42,
          metric_name: str | None = None,
      ) -> None: ...

      def optimize_and_train(
          self,
          X_train: Any,
          y_train: Any,
          X_val: Any,
          y_val: Any,
          n_trials: int = 50,
          study_name: str | None = None,
          timeout: float | None = None,
          n_jobs: int = 1,
          model_name: str | None = None,
          model_output_path: str | Path | None = None,
          show_progress_bar: bool = False,
      ) -> tuple[BaseModelWrapper, dict[str, Any], float]: ...

      def save_best_model(self, file_path: str | Path) -> None: ...
  ```
- **Lưu ý:** Yêu cầu tối thiểu một trial hoàn tất; nếu không có trial hoàn tất hoặc không có model tốt nhất, hàm raise lỗi. Mặc định dùng TPESampler với seed cấu hình và MedianPruner.

#### `model_factory.py` (`utils/ml_utils/model/model_factory.py`)

- **Vai trò:** Tạo regression estimators, gợi ý hyperparameters và bọc estimator bằng API fit/predict/persistence thống nhất.
- **Import chuẩn:** `from utils.ml_utils.model.model_factory import BaseModelWrapper, ModelFactory`
- **Public Interface (Stub):**
  ```python
  class BaseModelWrapper:
      def __init__(self, estimator: Any, model_name: str, pruning_metric: str = "rmse") -> None: ...
      def fit(
          self, X: Any, y: Any, X_val: Any = None, y_val: Any = None,
          trial: optuna.Trial | None = None, **fit_params: Any
      ) -> BaseModelWrapper: ...
      def predict(self, X: Any) -> Any: ...
      def predict_proba(self, X: Any) -> Any: ...
      def save(self, path: str | Path) -> None: ...
      @classmethod
      def load(cls, path: str | Path) -> BaseModelWrapper: ...

  class ModelFactory:
      @staticmethod
      def suggest_hyperparameters(trial: optuna.trial.BaseTrial, model_name: str | None = None) -> dict[str, Any]: ...
      @staticmethod
      def create_model(model_name: str, params: dict[str, Any] | None = None) -> BaseModelWrapper: ...
      @staticmethod
      def create_model_from_trial(
          trial: optuna.trial.BaseTrial, model_name: str | None = None
      ) -> tuple[BaseModelWrapper, str]: ...
  ```
- **Lưu ý:** Model aliases chuẩn hóa thành `ridge`, `random_forest`, `gradient_boosting` hoặc `xgboost`; model không hỗ trợ sẽ raise `ValueError`. XGBoost có callback pruning theo validation iterations; `predict_proba` không phù hợp với các regression estimators hiện tại.

#### `search_space.py` (`src/pipelines/ml_pipelines/calories_pred/search_space.py`)

- **Vai trò:** Khai báo search space Optuna theo điều kiện cho model type và các tham số tương ứng.
- **Import chuẩn:** `from src.pipelines.ml_pipelines.calories_pred.search_space import get_search_space`
- **Public Interface (Stub):**
  ```python
  def get_search_space(trial: optuna.trial.Trial) -> dict[str, object]:
      """Trả về model type và các hyperparameters được gợi ý cho trial."""
      ...
  ```
- **Lưu ý:** Được `hydra_config.yaml` tham chiếu làm `custom_search_space`; `model_train.py` không gọi hàm này trực tiếp trong luồng `RUN` hiện tại. File này chưa được khai báo trong `deps` của stage `calories_train`, nên DVC không tự phát hiện thay đổi nội dung của nó.

#### `dvc_handler.py` (`src/data_io/dvc_handler.py`)

- **Vai trò:** Adapter đọc dataset/model/metrics từ DVC và cung cấp retry/fallback thông qua decorator.
- **Import chuẩn:** `from src.data_io.dvc_handler import DVCHandler`
- **Public Interface (Stub):**
  ```python
  class DVCHandler:
      def __init__(self, repo: str | None = None, rev: str | None = None) -> None: ...
      def get_dataset(
          self, data_path: str, format_type: str = "csv",
          fallback_path: str | None = None,
      ) -> pd.DataFrame: ...
      def get_latest_model(self, model_path: str, framework: str = "sklearn") -> Any: ...
      def get_metrics(self, metrics_path: str = "metrics.json") -> DVCMetricsNormalizedSchema: ...
      def add_data(self, file_path: str) -> None: ...
  ```
- **Lưu ý:** `model_train.py` hiện đọc `.npy` và preprocessor trực tiếp bằng NumPy/joblib, không gọi `DVCHandler`; file vẫn được khai báo là dependency trong stage DVC. `add_data()` hiện là no-op có cảnh báo vì output được DVC pipeline quản lý.

#### `logger.py` (`src/nutrition_core/logging/logger.py`)

- **Vai trò:** Cấu hình logger chung cho console và các file log phân loại; cung cấp decorator đo thời gian và xử lý lỗi DVC/validation.
- **Import chuẩn:** `from src.nutrition_core.logging.logger import logger, monitor_performance, dvc_error_handler`
- **Public Interface (Stub):**
  ```python
  logger: logging.Logger

  def monitor_performance(time_limit: float = 5.0) -> Callable[..., Any]: ...
  def dvc_error_handler(
      max_retries: int = 3, retry_delay: int = 2, fallback_func: Callable[..., Any] | None = None
  ) -> Callable[..., Any]: ...
  def validation_error_handler() -> Callable[..., Any]: ...
  ```
- **Lưu ý:** Module tạo thư mục log dưới `logs/` khi import. Training entrypoint dùng `logger`; DVC handler dùng `dvc_error_handler`.

#### `exception.py` (`src/nutrition_core/exception/exception.py`)

- **Vai trò:** Định nghĩa các exception dùng chung cho data, ML, database và model training.
- **Import chuẩn:** `from src.nutrition_core.exception.exception import MLPipelineException, ModelTrainingError, ModelEvaluationError`
- **Public Interface (Stub):**
  ```python
  class NutritionBaseException(Exception): ...
  class DataPipelineException(NutritionBaseException): ...
  class MLPipelineException(NutritionBaseException): ...
  class ModelTrainingError(MLPipelineException): ...
  class ModelEvaluationError(MLPipelineException): ...
  ```
- **Lưu ý:** `model_train.py` và `ModelTrainer` hiện chưa raise các exception ML chuyên biệt này; chúng thường raise `ValueError`, `RuntimeError` hoặc lỗi thư viện. File được khai báo là dependency DVC.

## Ghi chú về khai báo dependency

Stage `calories_train` trong `setup_dvc_stages.sh` theo dõi `model_train.py`, `model_factory.py`, `model_trainer.py`, `dvc_handler.py`, `logger.py`, `exception.py` cùng artifacts đầu vào. Trong code đang chạy, `model_train.py` gọi `ModelTrainer` và logger; `ModelTrainer` gọi `ModelFactory`. `dvc_handler.py` và `exception.py` được liệt kê trong dependency của DVC nhưng không được import trực tiếp bởi entrypoint/trainer. `search_space.py` được tham chiếu trong cấu hình Hydra nhưng không nằm trong dependency list của stage. `mlflow_tracker.py` cũng không nằm trong dependency list; `ModelTrainer` ghi tracking trực tiếp qua MLflow API.
