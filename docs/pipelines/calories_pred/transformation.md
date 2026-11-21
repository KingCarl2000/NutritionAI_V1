# Calories Prediction — Data Transformation

## DVC stage `calories_transform`

Stage cấu hình trong [`dvc.yaml`](../../../dvc.yaml):

- **Lệnh chạy:** `python src/pipelines/ml_pipelines/calories_pred/data_trans.py`
- **Đầu vào và dependencies:** ba tập validated `train.csv`, `val.csv`, `test.csv`; validation report; drift report; cùng các module Python được liệt kê trong `deps`.
- **Tham số:** [`data_transformation.yaml`](../../../config/pipelines/ml_calories/data_transformation.yaml).
- **Đầu ra:** thư mục `Artifacts/data/ml_calories/transformed`.
- **Metrics:** `Artifacts/reports/ml_calories/transformation/transformation_report.json`, không cache (`cache: false`).

## Cấu hình YAML

### `data_transformation.yaml` (`config/pipelines/ml_calories/data_transformation.yaml`)

- **Vai trò:** Cấu hình đường dẫn artifacts, ánh xạ cột, preprocessing và MLflow cho bước biến đổi dữ liệu Calories.
- **Cách gọi trong code:** `CaloriesDataTransformation()` đọc YAML mặc định qua `DEFAULT_CONFIG_PATH` bằng `yaml.safe_load`.
- **Cây cấu trúc (Schema rút gọn):**
  ```yaml
  artifacts:
    root_dir: Artifacts/data/ml_calories/transformed
    valid_train_path: Artifacts/data/ml_calories/validated/train.csv
    valid_val_path: Artifacts/data/ml_calories/validated/val.csv
    valid_test_path: Artifacts/data/ml_calories/validated/test.csv
    valid_report_path: Artifacts/reports/ml_calories/validation/validation_report.json
    drift_report_path: Artifacts/reports/ml_calories/evidently/drift_report.json
    transformed_train_path: Artifacts/data/ml_calories/transformed/train.npy
    transformed_val_path: Artifacts/data/ml_calories/transformed/val.npy
    transformed_test_path: Artifacts/data/ml_calories/transformed/test.npy
    preprocessor_obj_path: Artifacts/data/ml_calories/transformed/preprocessing.pkl
    transformed_report_path: Artifacts/reports/ml_calories/transformation/transformation_report.json
  column_mapping:
    target: Calories
    id_column: User_ID
    numerical_features: [Age, Weight, Height, Duration, Heart_Rate, Body_Temp]
    categorical_features: [Gender]
  preprocessing:
    numerical:
      imputer: {type: knn, n_neighbors: 5, weights: uniform, strategy: median}
      scaler: {type: robust, with_centering: true, with_scaling: true, quantile_range: [25.0, 75.0]}
    categorical:
      imputer: {type: simple, strategy: most_frequent}
      encoder: {type: onehot, handle_unknown: ignore, sparse_output: false, drop: null}
  mlflow:
    experiment_name: calories_pred
    run_name: data_transformation
    log_params: true
    log_data_shapes: true
    register_preprocessor: true
  ```
- **Lưu ý:** Đường dẫn tương đối được phân giải từ thư mục gốc repo. Cả validation report và drift report phải qua gatekeeper; mỗi split phải có trạng thái `PASSED` và không có lỗi. Preprocessor chỉ fit trên `train`; mỗi file `.npy` chứa feature đã biến đổi cùng target.

## Module Python

### `data_trans.py` (`src/pipelines/ml_pipelines/calories_pred/data_trans.py`)

- **Vai trò:** Đọc dữ liệu validated, kiểm tra validation/drift gatekeeper, biến đổi ba split và ghi artifacts cùng lineage MLflow.
- **Import chuẩn:** `from src.pipelines.ml_pipelines.calories_pred.data_trans import CaloriesDataTransformation, transform_splits, verify_gatekeeper_reports`
- **Public Interface (Stub):**
  ```python
  PROJECT_ROOT: Path
  DEFAULT_CONFIG_PATH: Path
  EXPECTED_SPLITS: tuple[str, ...]

  def transform_splits(
      preprocessor: ColumnTransformer,
      feature_splits: Mapping[str, pd.DataFrame],
      target_splits: Mapping[str, pd.Series],
  ) -> dict[str, np.ndarray]:
      """Fit trên train, transform val/test và ghép target vào từng mảng."""
      ...

  def verify_gatekeeper_reports(
      validation_report_path: Path,
      drift_report_path: Path,
  ) -> dict[str, Any]:
      """Xác nhận validation và drift đạt yêu cầu; Raises: DataTransformationError."""
      ...

  class CaloriesDataTransformation:
      def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None: ...
      def run(self) -> dict[str, Any]:
          """Chạy transformation và trả về report; ghi các artifacts đã cấu hình."""
          ...
  ```
- **Lưu ý:** `run()` đọc các split qua `DVCHandler`, lưu `.npy`, `preprocessing.pkl` và transformation report; đồng thời log params, metrics và artifacts vào MLflow.

### `preprocessing.py` (`src/pipelines/ml_pipelines/calories_pred/preprocessing.py`)

- **Vai trò:** Tạo `ColumnTransformer` gồm pipeline xử lý feature số và feature phân loại theo cấu hình.
- **Import chuẩn:** `from src.pipelines.ml_pipelines.calories_pred.preprocessing import build_preprocessing_pipeline`
- **Public Interface (Stub):**
  ```python
  def build_preprocessing_pipeline(
      config: Mapping[str, Any],
      numerical_features: Sequence[str],
      categorical_features: Sequence[str],
  ) -> ColumnTransformer:
      """Tạo pipeline impute/scale feature số và impute/encode feature phân loại."""
      ...
  ```
- **Lưu ý:** Hỗ trợ imputer số `knn`/`simple`, scaler `robust`/`standard`/`minmax`/`none`, categorical imputer `simple`, encoder `onehot`/`ordinal`; loại cấu hình không hỗ trợ sẽ raise `ValueError`.

### `dvc_handler.py` (`src/data_io/dvc_handler.py`)

- **Vai trò:** Cung cấp adapter đọc dữ liệu qua DVC API và fallback về file local khi gặp lỗi remote.
- **Import chuẩn:** `from src.data_io.dvc_handler import DVCHandler`
- **Public Interface (Stub):**
  ```python
  class DVCHandler:
      def __init__(self, repo: str | None = None, rev: str | None = None) -> None: ...
      def get_dataset(
          self,
          data_path: str,
          format_type: str = "csv",
          fallback_path: str | None = None,
      ) -> pd.DataFrame:
          """Đọc CSV/Parquet qua DVC; fallback_path chỉ định file local dự phòng."""
          ...
  ```
- **Lưu ý:** Transformation truyền repo gốc cùng đường dẫn fallback của đúng split validated để không đọc nhầm dữ liệu raw.

### `exception.py` (`src/nutrition_core/exception/exception.py`)

- **Vai trò:** Định nghĩa exception dùng chung; transformation dùng `DataTransformationError` cho lỗi dữ liệu và gatekeeper.
- **Import chuẩn:** `from src.nutrition_core.exception.exception import DataTransformationError`
- **Public Interface (Stub):**
  ```python
  class DataPipelineException(NutritionBaseException): ...

  class DataTransformationError(DataPipelineException):
      """Lỗi phát sinh trong quá trình biến đổi dữ liệu."""
      ...
  ```