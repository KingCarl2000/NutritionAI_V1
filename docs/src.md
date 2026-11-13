# Tài liệu mã nguồn: Evidently và đánh giá Data Drift

Tài liệu này mô tả hai module tích hợp Evidently AI vào pipeline Calories Prediction: engine dùng chung ở `src/nutrition_core/pandera/evidently_engine.py` và evaluator của pipeline ở `src/pipelines/ml_pipelines/calories_pred/eval_drift.py`.

## 1. Evidently Drift Engine

*Mã nguồn:* [src/nutrition_core/pandera/evidently_engine.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/pandera/evidently_engine.py)

### `EvidentlyDriftEngine`

Engine điều phối kiểm định Pandera, tạo báo cáo và test suite Evidently, đồng bộ kết quả lên Evidently Workspace (nếu được cấu hình), và ghi metrics lên MLflow.

- **Khởi tạo** — `EvidentlyDriftEngine(artifact_dir, project_name, evidently_workspace_url=None, mlops_tracker=None)` lưu cấu hình và tracker, đồng thời tạo thư mục artifact nếu chưa tồn tại.
- **`validate_schema(df, schema_model)`** — Gọi `schema_model.validate(df)`. Trả về `(True, validated_df)` khi hợp lệ; nếu bắt gặp `pandera.errors.SchemaError`, ghi log và trả về `(False, None)`.
- **`run_drift_detection(current_df, reference_df, column_mapping_cfg, drift_options_cfg, **kwargs)`** — Xây dựng Evidently `DataDefinition` và hai `Dataset` từ DataFrame, sau đó chạy report trên các metric được cấu hình. Trả về dictionary gồm `report`, `snapshot` và `test_suite`.
  - **Data drift:** dùng `DataDriftPreset`, `DriftedColumnsCount`, metric `ValueDrift` cho các cột được chỉ định, cùng `EmbeddingsDrift` nếu có khai báo embeddings. Cấu hình mapping có thể xác định numerical, categorical, text, datetime, ID, target, prediction, task, positive label và embeddings. Các tùy chọn drift gồm ngưỡng tỷ lệ drift, phương pháp mặc định, p-value alpha và phương pháp theo nhóm/cột.
  - **Data quality:** luôn thêm metric tổng số giá trị thiếu và tương quan; có thể thêm kiểm tra giá trị thiếu, regex, range và danh sách hợp lệ/không hợp lệ cho các cột cấu hình.
  - **Đánh giá theo bài toán:** bổ sung preset classification hoặc regression khi task và target/prediction phù hợp; có tùy chọn dummy metrics. Hỗ trợ thêm Text/LLM evaluations và RecSys metrics qua các cấu hình tương ứng.
  - **Test suite CI/CD:** khi có `test_suite_options`, tạo Legacy `TestSuite` từ các preset drift, quality, stability, classification, regression, RecSys hoặc no-target-performance được bật; chạy suite và trả kết quả cùng report.
  - **Evidently Workspace:** khi có `evidently_workspace_url`, kết nối `RemoteWorkspace` với URL HTTP hoặc tạo `Workspace` cục bộ. Engine tìm hoặc tạo project, bổ sung các panel dashboard mặc định nếu chưa có, rồi đồng bộ snapshot và test suite. Lỗi đồng bộ được ghi log.
- **`_configure_workspace_dashboard(project, column_mapping_cfg)`** — Thêm panel đếm số dòng, tỷ lệ cột drift và biểu đồ drift theo các cột đã cấu hình (giới hạn tối đa 8 cột), tránh thêm panel trùng tên khi chạy lại.
- **`log_metrics_to_mlflow(report_json, test_suite_json=None)`** — Nếu tracker khả dụng, trích xuất metrics drift, missing values, chất lượng regression/classification và kết quả test suite; loại bỏ giá trị `None` trước khi gọi `tracker.log_metrics()`. Nếu không có tracker hoặc metric phù hợp, ghi log và bỏ qua.

## 2. Calories Prediction Drift Evaluation

*Mã nguồn:* [src/pipelines/ml_pipelines/calories_pred/eval_drift.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/pipelines/ml_pipelines/calories_pred/eval_drift.py)

### `CaloriesDriftEvaluator`

Lớp adapter cấp pipeline, chuyển cấu hình YAML sang định dạng mà `EvidentlyDriftEngine` sử dụng và quản lý việc xuất artifact cho mỗi lần đánh giá.

- **Khởi tạo** — `CaloriesDriftEvaluator(config, mlops_tracker=None)` lưu cấu hình/tracker và tạo engine. Các khóa trong `engine_config` là `report_dir` (mặc định `Artifacts/reports/ml_calories/data_drift`), `project_name` (mặc định `Calories_Prediction_Drift`) và `evidently_workspace_url`.
- **`_prepare_column_mapping()`** — Chuẩn hóa `column_mapping`: chấp nhận alias `datetime`/`datetime_columns` cho `datetime_features`; `user_id`, `item_id` hoặc `id` cho `id_column`; và `prediction_labels`, `prediction_label`, `prediction_probas` hoặc `prediction_proba` cho `prediction`. Các nhóm numerical, categorical, text và datetime được chuẩn hóa thành danh sách.
- **`_prepare_drift_options()`** — Chuyển `data_drift` (hoặc alias `drift_options`) thành các tùy chọn engine: tỷ lệ drift, phương pháp mặc định và p-value alpha; phương pháp/ngưỡng numerical và categorical; phương pháp/ngưỡng text; per-feature stattest; và danh sách cột cần kiểm tra drift. Có thể chuyển `per_column_options` thành mapping stattest theo cột. Lưu ý: ngưỡng riêng từng cột trong `per_column_options` hiện không được chuyển vào engine.
- **`execute_evaluation(current_df, reference_df)`** — Chạy toàn bộ luồng đánh giá:
  1. Chuẩn hóa column mapping và drift options, rồi gọi engine với các cấu hình `data_quality_options`, `regression_options` và `test_suite_options`.
  2. Nếu engine trả snapshot, xuất `drift_report.html`, `drift_report.json`, serialization đầy đủ `drift_snapshot.json` và `drift_metrics.csv` trong `engine.artifact_dir`. Các phương thức save/dump của snapshot được dùng theo thứ tự fallback để hỗ trợ các phiên bản API khác nhau. CSV được tạo từ danh sách metrics trong snapshot; nếu danh sách không có, ghi cảnh báo.
  3. Nếu có test suite, xuất `test_suite_report.html` và lấy dictionary kết quả.
  4. Gọi `engine.log_metrics_to_mlflow()` với report dictionary và test suite dictionary (nếu có), sau đó trả về toàn bộ kết quả của engine.

### Chạy module trực tiếp

Khi chạy file như script, module đọc `config/pipelines/ml_calories/evidently_profile.yaml`. Nếu không tìm thấy cấu hình, dùng cấu hình mẫu; sau đó tạo reference/current DataFrame mock và gọi `execute_evaluation()`. Đây là luồng thử nghiệm minh họa, không thay thế cấu hình và dữ liệu thực của pipeline.

## 3. Luồng tích hợp

`CaloriesDriftEvaluator` là điểm vào phù hợp cho pipeline Calories Prediction: nhận cấu hình pipeline và dữ liệu current/reference, chuẩn hóa các alias YAML, rồi ủy quyền phân tích cho `EvidentlyDriftEngine`. Engine sở hữu logic dựng report/test suite và tích hợp Workspace/MLflow; evaluator sở hữu việc xuất các artifact theo lần chạy. Schema validation Pandera được cung cấp riêng qua `validate_schema()` và cần được gọi khi pipeline muốn kiểm định schema trước bước phát hiện drift.
