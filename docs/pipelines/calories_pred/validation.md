# Tài liệu cấu hình kiểm định dữ liệu — Calories Prediction

Tóm tắt các YAML cấu hình và module liên quan đến kiểm định dữ liệu, theo dõi drift của pipeline `ml_calories`.

> **Cập nhật thay đổi gần đây:** Đối chiếu lịch sử Git từ **15 đến 17/11/2026**. Các cập nhật đã xác nhận tập trung ở schema Calories, luồng dùng `BaseSchema`, chuẩn hóa lỗi Pandera và xử lý Evidently. Các file `data_val.yaml`, `global.yaml` cùng các module logger, MLflow tracker, exception, custom checks, pandas backend, Pandera config và DVC handler không có commit thay đổi trong khoảng thời gian này.

## Cấu hình YAML

### `data_val.yaml` (`config/pipelines/ml_calories/data_val.yaml`)

- **Vai trò:** Khai báo schema train/inference, thư mục input/output/quarantine và ngưỡng lỗi riêng của validation.
- **Cách gọi trong code:** `DataValidationConfig(pipeline_config_path="config/pipelines/ml_calories/data_val.yaml")` trong `src/pipelines/ml_pipelines/calories_pred/data_val.py`.
- **Cấu trúc chính:** `schema_classes` trỏ đến `MlCaloriesTrainingSchema` và `MlCaloriesInputSchema`; `pipeline_paths` khai báo schema, Evidently profile, thư mục dữ liệu và báo cáo; `validation_params.error_threshold` mặc định là `0.05`.
- **Luồng hiện tại:** `DataValidationConfig` nạp cả global config và Evidently profile, lấy `info_threshold`/`warning_threshold` từ `global.yaml`, và dùng `warning_threshold` nếu không có `validation_params.error_threshold`. Validation tạo báo cáo JSON, lưu dataset hợp lệ, đưa toàn bộ input vào quarantine và dừng pipeline nếu error rate vượt ngưỡng. Nếu lỗi có index xác định nhưng nằm trong ngưỡng, các dòng lỗi bị loại khỏi dataset đầu ra.
- **Thay đổi gần đây:** Không thấy commit sửa YAML này trong khoảng thời gian đối chiếu. Tuy nhiên, `data_val.py` đã được đổi để schema kế thừa `BaseSchema` sử dụng chung triển khai validation và tracker MLflow hiện hành (xem phần module bên dưới).

### `evidently_profile.yaml` (`config/pipelines/ml_calories/evidently_profile.yaml`)

- **Vai trò:** Cấu hình Evidently cho báo cáo và phát hiện drift dữ liệu của pipeline Calories.
- **Cách gọi trong code:** `CaloriesDriftEvaluator(config=evidently_config, mlops_tracker=tracker)` trong `src/pipelines/ml_pipelines/calories_pred/data_val.py`.
- **Cấu trúc chính hiện tại:**
  ```yaml
  engine_config:
    project_name: "Calories_Prediction_Drift_Monitoring"
    reference_data: "D:/NutritionAI_V1/Artifacts/data/ml_calories/training/train.csv"
    current_data: "D:/NutritionAI_V1/Artifacts/data/ml_calories/validated/train.csv"
    report_dir: "D:/NutritionAI_V1/Artifacts/reports/ml_calories/evidently"
    evidently_workspace_url: null
  column_mapping:
    target: "Calories"
    id_column: "User_ID"
    prediction: "predicted_calories"
    task: "regression"
    numerical_features: [Age, Weight, Height, Duration, Heart_Rate, Body_Temp]
    categorical_features: [Gender]
  data_drift:
    dataset_drift_share: 0.5
    num_stattest: "wasserstein"
    num_stattest_threshold: 0.1
    cat_stattest: "jensenshannon"
    cat_stattest_threshold: 0.1
    per_column_options: <object>
  ```
- **Thay đổi gần đây:** Ánh xạ cột được đổi sang đúng kiểu chữ của dataset (`Calories`, `Age`, `Gender`...), và cấu hình bỏ `datetime: created_at`. Các đường dẫn dữ liệu/báo cáo hiện là đường dẫn tuyệt đối trên `D:` nên cần đổi khi chạy ở môi trường khác.
- **Lưu ý:** `CaloriesDriftEvaluator` chuẩn hóa alias và danh sách cột trước khi gọi engine. `per_column_options` được chuyển thành mapping phép kiểm định theo cột; dù YAML có `threshold` riêng từng cột, phần chuẩn hóa hiện chỉ truyền `stattest` cho mapping này. Engine hiện tạo `DataDriftPreset(columns=...)`; các tham số drift chuẩn hóa không được truyền vào preset trong phiên bản code hiện tại.

### `global.yaml` (`config/pipelines/global.yaml`)

- **Vai trò:** Cấu hình dùng chung cho các pipeline, gồm API lõi, MLflow và ngưỡng MLOps.
- **Cấu trúc chính:** `core_api_references`, `mlflow_config` và `mlops_thresholds`; `data_failure_rate` gồm `info_threshold: 0.01` và `warning_threshold: 0.05`.
- **Thay đổi gần đây:** Không thấy commit sửa file này trong khoảng thời gian đối chiếu.

## Module Python liên quan

### `calories_schema.py` (`config/pipelines/ml_calories/calories_schema.py`)

- **Vai trò:** Khai báo schema input và training riêng cho Calories.
- **Thay đổi gần đây:** `MlCaloriesInputSchema` hiện kế thừa `BaseSchema`; `Config` kế thừa `BaseSchema.Config` nhưng đặt `coerce=True`, `strict=True`. Các cột sử dụng custom checks đã đăng ký từ `src.nutrition_core.pandera.api.check`: `User_ID` từ 1 đến `2**63 - 1`, `Gender` thuộc `male/female/other` và độ dài 1–10, `Age` 0–200, `Height` 40–255, `Weight` 2–300, `Duration` 0–300, `Heart_Rate` 40–220, `Body_Temp` 35–42.5. `MlCaloriesTrainingSchema` kế thừa input schema và thêm `Calories` 1–5000.

### `data_val.py` (`src/pipelines/ml_pipelines/calories_pred/data_val.py`)

- **Vai trò:** Nạp cấu hình/schema, chạy Pandera validation, tạo báo cáo lỗi, quarantine dữ liệu và gọi drift evaluator.
- **Thay đổi gần đây:** `validate_with_pandas_backend(...)` nhận `tracker: Optional[MLOpsTracker]`. Nếu schema kế thừa `BaseSchema`, hàm gọi `schema_class.validate_df(..., mlflow_tracker=tracker, return_errors=True)`; schema khác vẫn dùng `NutritionPandasBackend` trực tiếp. `DataValidationError` được giữ nguyên thay vì bọc lại thành lỗi backend chung. Pipeline truyền tracker đang chạy vào validation, áp dụng `setup_pandera_validation` trong MLflow run, và sao chép `validation_report_train.json` sang artifact report được DVC theo dõi.
- **Public interface liên quan:**
  ```python
  def validate_with_pandas_backend(
      df: pd.DataFrame,
      schema_class: Type[Any],
      lazy: bool = True,
      tracker: Optional[MLOpsTracker] = None,
  ) -> Tuple[pd.DataFrame, List[dict]]: ...
  ```

### `logger.py` (`src/nutrition_core/logging/logger.py`)

- **Vai trò:** Khởi tạo logger dùng chung; cung cấp decorator đo hiệu năng, xử lý lỗi DVC và lỗi validation.
- **Giao diện liên quan:** `logger`, các hằng đường dẫn log, `monitor_performance`, `dvc_error_handler`, `validation_error_handler`.
- **Thay đổi gần đây:** Không thấy commit sửa file này trong khoảng thời gian đối chiếu.

### `mlflow_tracker.py` (`src/nutrition_core/logging/mlflow_tracker.py`)

- **Vai trò:** Bao bọc thao tác MLflow cho run, metrics, lineage, model, artifacts, system logs và validation errors.
- **Giao diện liên quan:** `MLOpsTracker` với các phương thức `start_run`, `log_params_and_metrics`, `log_error_and_tag`, `log_dataset_lineage`, `log_model`, `log_validation_errors`, `end_run`, `log_artifact_file`.
- **Thay đổi gần đây:** Không thấy commit sửa file này trong khoảng thời gian đối chiếu.

### `exception.py` (`src/nutrition_core/exception/exception.py`)

- **Vai trò:** Ngoại lệ dùng chung cho pipeline dữ liệu, ML và cơ sở dữ liệu; bổ sung thông tin file/dòng từ traceback.
- **Giao diện liên quan:** `NutritionBaseException` và các nhóm `DataPipelineException`, `MLPipelineException`, `DatabaseException` cùng các lớp con.
- **Thay đổi gần đây:** Không thấy commit sửa file này trong khoảng thời gian đối chiếu.

### `val_exception.py` (`src/nutrition_core/exception/val_exception.py`)

- **Vai trò:** Chuẩn hóa lỗi Pandera thành `DataValidationError` và trích xuất chi tiết eager/lazy validation.
- **Giao diện:** `DataValidationError(message, errors)`; `PanderaErrorHandler.parse_lazy_errors`, `parse_eager_error`, `handle_validation_error`.
- **Thay đổi gần đây:** Không thấy commit sửa file này trong khoảng thời gian đối chiếu. Các decorator hiện gọi `handle_validation_error` để chuyển `SchemaError`/`SchemaErrors` thành ngoại lệ chuẩn.

### `base.py` (`src/nutrition_core/pandera/api/base.py`)

- **Vai trò:** Schema Pandera dùng chung và phương thức validation DataFrame qua `NutritionPandasBackend`, có thể ghi kết quả/lỗi lên MLflow.
- **Thay đổi gần đây:** `BaseSchema.validate_df` không còn khởi tạo tracker MLflow toàn cục. Tracker được truyền tùy chọn từ caller; thêm `return_errors` để caller xử lý routing/ngưỡng riêng. Khi có lỗi, phương thức log lỗi và trạng thái/metrics nếu có tracker; nếu `drop_invalid_rows=False` và `return_errors=False` thì ném `DataValidationError`. Khi `return_errors=True`, trả về cặp `(validated_df, errors)`.
- **Public interface cập nhật:**
  ```python
  class BaseSchema(pa.DataFrameModel):
      class Config:
          coerce = True
          strict = False
          ordered = False
          unique_column_names = True
          add_missing_columns = True

      @classmethod
      def validate_df(
          cls,
          df: pd.DataFrame,
          lazy: bool = True,
          drop_invalid_rows: bool = False,
          mlflow_tracker: Optional[MLOpsTracker] = None,
          return_errors: bool = False,
      ) -> Any: ...
  ```

### `check.py` (`src/nutrition_core/pandera/api/check.py`)

- **Vai trò:** Custom check vectorized và factory tạo quy tắc kiểm định đa cột, theo nhóm, regex và cảnh báo outlier.
- **Giao diện liên quan:** `is_in_numeric_range`, `is_valid_string_length`, `GenericCheckFactory`.
- **Thay đổi gần đây:** Không thấy commit sửa file này trong khoảng thời gian đối chiếu; schema Calories hiện import module này để đăng ký các Field checks.

### `pandas_executor.py` (`src/nutrition_core/pandera/backends/pandas_executor.py`)

- **Vai trò:** Thực thi validation trên `pandas.DataFrame`, gồm kiểm tra đầu vào, ép kiểu, gom lỗi và tùy chọn loại bỏ dòng lỗi.
- **Giao diện liên quan:** `NutritionPandasBackend` và `validate_with_pandas_backend`.
- **Thay đổi gần đây:** Không thấy commit sửa file này trong khoảng thời gian đối chiếu. `BaseSchema` vẫn sử dụng backend này làm implementation chung.

### `config.py` (`src/nutrition_core/pandera/config.py`)

- **Vai trò:** Thiết lập validation depth Pandera từ `PANDERA_VALIDATION_DEPTH` và tùy chọn ghi cấu hình lên MLflow.
- **Giao diện liên quan:** `setup_pandera_validation(mlflow_tracker=None)`; hỗ trợ `SCHEMA_ONLY`, `DATA_ONLY`, `SCHEMA_AND_DATA`.
- **Thay đổi gần đây:** Không thấy commit sửa file này trong khoảng thời gian đối chiếu. Pipeline Calories tiếp tục mở context này trong MLflow run validation/drift.

### `decorators.py` (`src/nutrition_core/pandera/decorators.py`)

- **Vai trò:** Bọc `pandera.check_types` và `pandera.check_io`, ghi log và chuẩn hóa validation exception.
- **Thay đổi gần đây:** Thêm `_normalize_validation_error`, dùng `PanderaErrorHandler.handle_validation_error(...)` thay cho lời gọi `extract_errors` không tương thích. Cả hai decorator bắt riêng `SchemaError`/`SchemaErrors`, log lỗi, rồi ném `DataValidationError` có giữ nguyên exception gốc làm cause. Lỗi ngoài nhóm Pandera được log bằng `logger.exception` rồi ném lại.
- **Giao diện:** `nutrition_check_types(*args, **kwargs)` và `nutrition_check_io(head=5, tail=5, **kwargs)`.

### `evidently_engine.py` (`src/nutrition_core/pandera/evidently_engine.py`)

- **Vai trò:** Engine phát hiện drift, kiểm tra chất lượng, quản lý Evidently Workspace và ghi metrics MLflow.
- **Thay đổi gần đây:** Engine tính giao các cột có mặt trong cả current/reference, cảnh báo và bỏ qua cột cấu hình bị thiếu; chỉ bật regression/classification evaluation khi các cột cần thiết sẵn có. Target và prediction dạng regression được thêm vào numerical features khi có trong dữ liệu. Có thể chọn cột drift qua `selected_drift_cols`; test suite nhận DataFrame gốc và `additional_data` chỉ khi đó là dict. Đồng bộ test-suite run lên Workspace chỉ khi client hỗ trợ `add_test_run`; MLflow logging kiểm tra `log_metrics` có tồn tại và bắt lỗi khi ghi.
- **Lưu ý:** Engine nhận `timestamp`, `timestamp_column` hoặc `time_col` làm timestamp mapping. `CaloriesDriftEvaluator` hiện chủ động đặt `datetime_features=[]`, do đó không đưa cột datetime vào DataDefinition cho pipeline này. Engine có log shape, tên cột và dtype của hai DataFrame trước khi chạy report.

### `dvc_handler.py` (`src/data_io/dvc_handler.py`)

- **Vai trò:** Adapter DVC để đồng bộ dữ liệu, tải dataset/mô hình/metrics/ảnh và chuẩn hóa metrics bằng Pydantic.
- **Giao diện liên quan:** `DVCHandler`, `DVCMetricsNormalizedSchema` và các model metric; hỗ trợ dataset CSV/Parquet.
- **Thay đổi gần đây:** Không thấy commit sửa file này trong khoảng thời gian đối chiếu.

### `eval_drift.py` (`src/pipelines/ml_pipelines/calories_pred/eval_drift.py`)

- **Vai trò:** Chuẩn hóa cấu hình drift của Calories, gọi engine và xuất artifacts/report/metrics.
- **Thay đổi gần đây:** `_prepare_column_mapping` luôn đặt `datetime_features=[]`, chuẩn hóa các feature thành list và hỗ trợ alias cho ID/prediction. Khi xuất kết quả, HTML, JSON và serialization đầy đủ được tạo từ cùng snapshot; ưu tiên các API có sẵn trên snapshot (`save_html`, `save_json`, `save`, `dumps`, `dump_dict`...) với phương án dự phòng. CSV drift metrics được tạo bằng cách flatten danh sách metrics trong snapshot; test suite report HTML được lưu nếu có.
- **Giao diện:** `CaloriesDriftEvaluator(config, mlops_tracker=None)` và `execute_evaluation(current_df, reference_df)`; các tham số vẫn tương thích với stub đã nêu trước đây.

## Phạm vi kiểm tra lịch sử

Các thay đổi liên quan đã xác nhận qua các commit ngày 15–16/11/2026, gồm refactor dùng `BaseSchema` cho schema Calories, cập nhật luồng validation, chuẩn hóa lỗi decorator, và thay đổi cấu hình/xử lý Evidently. Những file/module được ghi “không thấy commit sửa” ở trên không có thay đổi trực tiếp theo lịch sử commit đã đối chiếu trong cửa sổ này.