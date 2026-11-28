# Calories Prediction — BentoML Packaging

## Stage 5: Build and package the BentoML prediction service

Stage `calories_package` lấy các inference pipeline đã lưu từ training, đăng ký chúng vào BentoML Model Store, rồi build service thành Bento bundle. DVC quản lý cả Model Store và bundle tại `Artifacts/data/ml_calories/bentoml`; stage không cần truy cập MLflow để lấy model.

Stage phụ thuộc vào training report, ba pipeline theo metric (`mean_squared_error`, `r2_score`, `root_mean_squared_error`), `service.py`, cấu hình Bento và các dependency manifests. Service hiện phục vụ model `r2_score`.

```bash
bash src/scripts/setup_dvc_stages.sh
dvc repro calories_package
```

Sau khi build, đặt `BENTOML_HOME` trỏ đến `Artifacts/data/ml_calories/bentoml`, rồi chạy `bentoml serve calories-prediction:latest --port 3000`. Để tạo container image, chạy `bentoml containerize calories-prediction:latest` với cùng `BENTOML_HOME` và Docker daemon đang hoạt động.

## Cấu hình YAML

### `dvc.yaml` (`dvc.yaml`)
- **Vai trò:** Khai báo dependencies, lệnh chạy và output DVC của stage đóng gói Calories.
- **Cách gọi trong code:** `dvc repro calories_package`
- **Cây cấu trúc (Schema rút gọn):**
  ```yaml
  stages:
    calories_package:
      cmd: python src/pipelines/ml_pipelines/calories_pred/model_package.py
      deps: [<training pipelines>, <training report>, <service/config/dependencies>]
      outs:
        - Artifacts/data/ml_calories/bentoml
  ```
- **Lưu ý:** Ba pipeline `.pkl` và training report được tạo bởi stage `calories_train`; thay đổi dependencies sẽ khiến DVC chạy lại stage packaging.

### `bentofile.yaml` (`config/pipelines/ml_calories/bentofile.yaml`)
- **Vai trò:** Chỉ định Bento service entrypoint và file mã nguồn cần đưa vào Bento bundle.
- **Cách gọi trong code:** `bentoml build --name calories-prediction --bentofile config/pipelines/ml_calories/bentofile.yaml`
- **Cây cấu trúc (Schema rút gọn):**
  ```yaml
  service: "service:CaloriesPredictionService"
  include:
    - "service.py"
  ```

### `train_params.yaml` (`config/pipelines/ml_calories/train_params.yaml`)
- **Vai trò:** Cấu hình metric huấn luyện và đường dẫn artifacts mà stage packaging nhận từ stage training.
- **Cách gọi trong code:** Hydra merge file này vào `cfg.train_params`; `model_train.py` đọc `cfg.train_params.artifacts` và `cfg.train_params.metric_names`.
- **Cây cấu trúc (Schema rút gọn):**
  ```yaml
  train_params:
    metric_names: [mean_squared_error, r2_score, root_mean_squared_error]
    artifacts:
      best_model_path: Artifacts/data/ml_calories/models/best_model.pkl
      training_report_path: Artifacts/reports/ml_calories/training/training_report.json
    mlflow:
      tracking_uri: "${oc.env:MLFLOW_TRACKING_URI,sqlite:///mlflow.db}"
  ```
- **Lưu ý:** Với nhiều metric, `model_train.py` thêm metric vào tên model và lưu pipeline inference cạnh model tương ứng; đường dẫn được ghi trong training report.

## Module Python

### `model_train.py` (`src/pipelines/ml_pipelines/calories_pred/model_train.py`)
- **Vai trò:** Huấn luyện model, ghép preprocessor với estimator thành inference pipeline và ghi đường dẫn pipeline vào training report.
- **Import chuẩn:** `from src.pipelines.ml_pipelines.calories_pred.model_train import run_training`
- **Public Interface (Stub):**
  ```python
  PROJECT_ROOT: Path
  CONFIG_PATH: str
  METRICS: dict[str, tuple[Callable[..., float], str]]

  def run_training(cfg: DictConfig) -> dict[str, Any]:
      """Train các metric đã cấu hình và ghi model, inference pipeline, report."""
      ...

  def main(cfg: DictConfig) -> None:
      """Hydra entrypoint gọi run_training(cfg)."""
      ...
  ```
- **Lưu ý:** Stage packaging tiêu thụ các pipeline `.pkl` và training report do module này tạo.

### `model_register.py` (`src/pipelines/ml_pipelines/calories_pred/model_register.py`)
- **Vai trò:** Đọc training report, nạp inference pipeline đã lưu và đăng ký từng metric vào BentoML Model Store.
- **Import chuẩn:** `from src.pipelines.ml_pipelines.calories_pred.model_register import register_models_from_report`
- **Public Interface (Stub):**
  ```python
  PROJECT_ROOT: Path
  DEFAULT_REPORT_PATH: Path

  def register_models_from_report(
      report_path: str | Path = DEFAULT_REPORT_PATH,
      model_name_prefix: str = "calories_predictor",
  ) -> dict[str, Any]:
      """Đăng ký các pipeline preprocessing + model và trả về model theo metric."""
      ...

  def main() -> None:
      """Đăng ký models từ report được truyền qua CLI."""
      ...
  ```
- **Lưu ý:** Mỗi metric phải có `inference_pipeline_path` hợp lệ trong training report; đường dẫn tương đối được phân giải từ project root.

### `model_package.py` (`src/pipelines/ml_pipelines/calories_pred/model_package.py`)
- **Vai trò:** Cấu hình BentoML Model Store, gọi đăng ký model và build Bento bundle.
- **Import chuẩn:** `from src.pipelines.ml_pipelines.calories_pred.model_package import build_calories_service`
- **Public Interface (Stub):**
  ```python
  PROJECT_ROOT: Path
  DEFAULT_BENTOML_HOME: Path
  BENTOFILE_PATH: Path

  def build_calories_service(
      report_path: str | Path = DEFAULT_REPORT_PATH,
      bentoml_home: str | Path = DEFAULT_BENTOML_HOME,
  ) -> None:
      """Đăng ký các pipeline và build Bento service bundle."""
      ...

  def main() -> None:
      """Đọc CLI arguments và chạy packaging."""
      ...
  ```
- **Lưu ý:** Cần BentoML CLI trong môi trường đang chạy. `BENTOML_HOME` được đặt cho cả bước đăng ký model và tiến trình `bentoml build`; lỗi build sẽ dừng stage.

### `service.py` (`service.py`)
- **Vai trò:** Khai báo HTTP service nhận features thô, kiểm tra request và trả về calories prediction.
- **Import chuẩn:** `from service import CaloriesPredictionRequest, CaloriesPredictionResponse, CaloriesPredictionService`
- **Public Interface (Stub):**
  ```python
  MODEL_TAG: str
  runtime_image: bentoml.images.Image
  NUMERICAL_FEATURES: tuple[str, ...]
  FEATURES: tuple[str, ...]

  class CaloriesPredictionRequest(BaseModel):
      Gender: Literal["male", "female", "other"]
      Age: int = Field(ge=0, le=200)
      Weight: float = Field(ge=2, le=300)
      Height: float = Field(ge=40, le=255)
      Duration: float = Field(ge=0, le=300)
      Heart_Rate: float = Field(ge=40, le=220)
      Body_Temp: float = Field(ge=35, le=42.5)

  class CaloriesPredictionResponse(BaseModel):
      Calories: float

  def _raw_features(
      request: CaloriesPredictionRequest,
  ) -> pd.DataFrame:
      """Chuyển request đã validate thành DataFrame theo schema model."""
      ...

  class CaloriesPredictionService:
      def __init__(self) -> None: ...
      def predict(
          self, request: CaloriesPredictionRequest
      ) -> CaloriesPredictionResponse:
          """Dự đoán calories và kiểm tra output hữu hạn."""
          ...
  ```
- **Lưu ý:** Service tải `calories_predictor_r2_score:latest` từ BentoML Model Store; runtime image khai báo Python và các package inference.