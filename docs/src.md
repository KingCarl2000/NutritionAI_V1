# DEVELOPER GUIDE: NutritionAI_V1 Core Modules

Tài liệu này tổng hợp danh sách các class, hàm và decorator thuộc phân hệ `src`, giúp nhà phát triển dễ dàng tra cứu và tái sử dụng khi mở rộng hệ thống.

## 1. Logging & MLOps Tracking

**File:** `src/nutrition_core/logging/logger.py`

* **`logger` (Instance):** Đối tượng logging chính của hệ thống.
* **`monitor_performance(time_limit=5.0)`:** Decorator giám sát thời gian thực thi của hàm.

**File:** `src/nutrition_core/logging/mlflow_tracker.py`

* **`MLOpsTracker`:** Lớp quản lý MLflow Tracking, tự động ghi nhận system metrics.
* **`start_run`, `upload_system_logs`, `log_params_and_metrics`, `log_error_and_tag`:** Các phương thức thao tác vòng đời MLflow.

## 2. Data Input/Output (`src/data_io`)

Phân hệ chịu trách nhiệm trích xuất, biến đổi, nạp dữ liệu (ETL) và quản lý phiên bản dữ liệu/metrics.

*File:* [base_extractor.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/data_io/base_extractor.py)

- *BasePostgresExtractor* *(ABC)*: Lớp cơ sở trừu tượng quản lý kết nối và trích xuất dữ liệu từ PostgreSQL. Cung cấp phương thức `_convert_data_types()` để phẳng hóa (flatten) các trường dữ liệu JSON/JSONB lồng nhau thành các cột Pandas DataFrame riêng biệt.

*File:* [base_trans.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/data_io/base_trans.py)

- *BaseTransformer* *(ABC)*: Lớp cơ sở cho các tác vụ Data Transformation (làm sạch và chuẩn hóa). Tích hợp đối tượng `MLOpsTracker` để theo dõi. Cung cấp hàm `fit_transform()` được gắn decorator `@monitor_performance` để tự động đo lường thời gian thực thi và đồng bộ lên MLflow.

*File:* [bulk_loader.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/data_io/bulk_loader.py)

- *PostgresBulkLoader*: Lớp quản lý nạp dữ liệu lớn. Hàm cốt lõi `execute_bulk_load()` sử dụng lệnh `COPY` của PostgreSQL để nạp dữ liệu từ file (CSV/TEXT) vào database tốc độ cao, đồng thời kiểm soát Transaction (Commit/Rollback) và ghi nhận thông số tracking qua MLflow.

*File:* [dvc_handler.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/data_io/dvc_handler.py)

- *Pydantic Metrics Schemas*: Định nghĩa các cấu trúc chuẩn hóa cho chỉ số đánh giá (ví dụ: `GlobalSummaryMetrics`, `EpochMetrics`, `DVCMetricsNormalizedSchema`) để đảm bảo tính nhất quán.
- *DVCHandler*: Lớp Adapter ẩn đi sự phức tạp của DVC, cung cấp giao diện chuẩn để tương tác, ghi nhận và quản lý phiên bản của các metrics/metadata trong quá trình huấn luyện mô hình.

## 3. Quản Lý Ngoại Lệ (Exception Handling)

**File:** `src/nutrition_core/exception/exception.py` & `val_exception.py`

* **`NutritionBaseException`**: Ngoại lệ gốc, tự động bắt Traceback.
* **`DataPipelineException`, `DataValidationError`**: Các ngoại lệ cho Pipeline và Pandera validation.

## 4. Tương Tác PostgreSQL (Core & Monitoring)

**File:** `src/postgres/core/connection.py`

* Thiết lập kết nối cơ sở dữ liệu sử dụng `sqlalchemy.create_engine` và thư viện `psycopg` cùng `psycopg_pool.ConnectionPool` để tối ưu hóa pool kết nối.

**File:** `src/postgres/monitoring/query_analyzer.py`

* **`analyze_query_performance(query: str)`:** Hàm tự động chèn cú pháp `EXPLAIN (ANALYZE, FORMAT JSON)` vào truy vấn.

## 5. Feature Store Engine

**File:** `src/postgres/feature_store/engine.py`

* **`FeatureStoreEngine(conn_info)`:** Khởi tạo engine quản lý kho đặc trưng với chuỗi kết nối PostgreSQL.
* **`copy_from_csv(table_name, file_path)`:** Nạp dữ liệu trực tiếp từ file CSV phía Client lên máy chủ thông qua giao thức `COPY FROM STDIN`.

## 6. ML Pipelines (Data Ingestion)

### 1. Calories Prediction ( src/pipelines/ml_pipelines/calories_pred )

Phân hệ này quản lý vòng đời dữ liệu đầu vào, kiểm định chất lượng và giám sát mô hình cho dự án NutritionAI_V1.

*File:* [data_ingest.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/pipelines/ml_pipelines/calories_pred/data_ingest.py)
- **`CaloriesDataIngestion`**: Lớp kế thừa từ `BasePostgresExtractor`, chịu trách nhiệm trích xuất dữ liệu trực tiếp từ PostgreSQL. Đọc linh hoạt tên bảng, lược đồ và danh sách cột loại bỏ (drop) từ cấu hình YAML thay vì hardcode. Tích hợp sẵn `DVCHandler` và `MLOpsTracker` để theo dõi dữ liệu, tự động chia tách tập Train/Test/Validation.

*File:* [data_val.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/pipelines/ml_pipelines/calories_pred/data_val.py)
- **`CaloriesDataValidationPipeline`**: Lớp điều phối quá trình kiểm định chất lượng dữ liệu. Sử dụng hàm `load_class_from_registry()` để nạp động các validation schema từ cấu hình YAML.
- **`run_validation_pipeline()`**: Hàm thực thi kiểm định thông qua Pandera. Tự động định tuyến các bản ghi lỗi vào thư mục Quarantine, phát cảnh báo (Alert) và dừng pipeline nếu tỷ lệ lỗi vượt ngưỡng cho phép, đồng thời ghi nhận metrics lên MLflow.

*File:* [eval_drift.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/pipelines/ml_pipelines/calories_pred/eval_drift.py)
- **`CaloriesDriftEvaluator`**: Lớp đánh giá Data Drift (trôi lệch dữ liệu) tích hợp Evidently. Khởi tạo và trích xuất động các cấu hình đường dẫn và ngưỡng cảnh báo (như PSI warning, PSI critical, p-value alpha) từ `evidently_profile.yaml` và `global.yaml`.
- **`evaluate_drift()`**: Phương thức đo lường độ trôi lệch của tập dữ liệu mới so với tập tham chiếu (reference). Được giám sát thời gian thực thi bởi decorator `@monitor_performance`.

## 7. Data Validation (Hệ sinh thái Pandera & evidently AI)

**File:** [`src/nutrition_core/pandera/api/base.py`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/pandera/api/base.py)

* **`BaseSchema`:** Lớp Schema cơ sở kế thừa từ `pa.DataFrameModel`. Lớp này áp dụng các quy tắc kiểm định khung cho tất cả mô hình dữ liệu về Nutrition. Mọi Schema lớp con (như `CaloriesSchema`, `UserSchema`) đều nên kế thừa lớp này.

**File:** [`src/nutrition_core/pandera/api/check.py`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/pandera/api/check.py)

* **`is_in_numeric_range(pandas_obj, min_val, max_val)`:** Phương thức kiểm tra vectorized mở rộng (được đăng ký qua `@pa_ext.register_check_method`), dùng để kiểm tra một cột số có nằm trong khoảng `[min_val, max_val]` hay không.
* **`is_valid_string_length(pandas_obj, min_len, max_len)`:** Phương thức kiểm tra vectorized dùng để xác thực độ dài chuỗi ký tự nằm trong khoảng `[min_len, max_len]`.

**File:** [`src/nutrition_core/pandera/backends/pandas_executor.py`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/pandera/backends/pandas_executor.py)

* **`NutritionPandasBackend`:** Backend chuyên biệt hoạt động như một lớp trung gian (Facade) để xử lý dữ liệu Pandas DataFrame. Nó chịu trách nhiệm phân giải Schema, duyệt dữ liệu, tiền xử lý, thực thi các vectorized checks và gom nhóm lỗi.
* **`resolve_backend(data)`:** Class method xác thực dữ liệu đầu vào bắt buộc phải là `pandas.DataFrame`, nếu không sẽ ném ra `TypeError`.

**File:** [`src/nutrition_core/pandera/decorators.py`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/pandera/decorators.py)

* **`nutrition_check_types`:** Decorator tự động kiểm tra dữ liệu dựa trên Type Annotations của hàm.
* **`_log_validation_failure`:** Hàm nội bộ gọi hook của `mlflow_tracker` (`log_validation_failure`) để ghi nhận lỗi xác thực trực tiếp lên hệ thống tracking khi validation thất bại.

**File:** [`src/nutrition_core/pandera/config.py`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/pandera/config.py)

* Chứa các cấu hình toàn cục (global configuration) cho toàn bộ tiến trình kiểm định dữ liệu.

*File:* [`src/nutrition_core/pandera/evidently_engine.py`]
- `EvidentlyDriftEngine` (Class): Engine lõi tích hợp Pandera Schema Validation, Evidently AI Data Drift, quản lý Artifact bằng DVC và MLflow.
- `validate_schema(df, schema_model)`: Thực thi kiểm tra lược đồ dữ liệu động (Pandera Schema Validation) [Giai đoạn 1].
- `run_drift_detection(current_df, reference_df, ...)`: Phát hiện trôi dạt dữ liệu bằng Evidently AI dựa trên cấu hình (dataset_drift_share, method, thresholds...), đồng thời tạo snapshot và báo cáo HTML/JSON [Giai đoạn 2].

## 8. Database & CRUD ( src/nutrition_core/database )
Phân hệ quản lý kết nối cơ sở dữ liệu, khai báo mô hình (ORM) và các thao tác truy vấn (CRUD).

*File:* [`src/nutrition_core/database/session.py`]
- Cung cấp `Base` (declarative_base) và quản lý session của SQLAlchemy.
- Thiết lập connection pool để tương tác với cơ sở dữ liệu PostgreSQL.

*File:* [`src/nutrition_core/database/models.py`]
- `build_model_from_yaml(yaml_filename)`: Đọc tệp cấu hình YAML (từ `config/datasets/`) và tự động đăng ký Model động vào `Base.metadata`. Hàm này hỗ trợ ánh xạ tự động các kiểu dữ liệu cơ bản (integer, string, float, date...) từ cấu hình YAML sang SQLAlchemy Types.

*Thư mục:* [`src/nutrition_core/database/crud/`]
- Chứa các module xử lý thao tác Create, Read, Update, Delete.
- Gồm các tệp chuyên biệt như `user_crud.py` (quản lý người dùng) và `vector_crud.py` (quản lý cơ sở dữ liệu vector/RAG).


## 9. API Framework ( src/api )

