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

**File:** `src/pipelines/ml_pipelines/calories_pred/data_ingest.py`

* **`CaloriesDataIngestion(BasePostgresExtractor)`:** Class phụ trách trích xuất dữ liệu, được tích hợp sẵn với `DVCHandler` và `MLOpsTracker`.
* **`extract_data_from_postgresql()`:** Hàm trích xuất dữ liệu thô.
* **`process_and_split_data(df)`:** Bước tiền xử lý, tự động loại bỏ các đặc trưng không cần thiết.

## 7. Data Validation (Hệ sinh thái Pandera)

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
