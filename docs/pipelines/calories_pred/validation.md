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
