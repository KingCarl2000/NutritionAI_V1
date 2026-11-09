# DEVELOPER GUIDE: NutritionAI_V1 Core Modules

Tài liệu này tổng hợp danh sách các class, hàm và decorator thuộc phân hệ `src`, giúp nhà phát triển dễ dàng tra cứu và tái sử dụng khi mở rộng hệ thống.

## 1. Logging & MLOps Tracking

Phân hệ này chịu trách nhiệm quản lý nhật ký hệ thống (system logs) nhiều cấp độ và tự động hóa việc theo dõi vòng đời dữ liệu/mô hình thông qua MLflow.

### A. System Logging (Ghi nhận nhật ký hệ thống)

*Mã nguồn:* [src/nutrition_core/logging/logger.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/logging/logger.py)

- **`logger` *(Instance)***: Đối tượng logging lõi của dự án (`NutritionAI`). Tự động phân luồng nhật ký theo thời gian thực (`logs/{thư_mục}/MM_DD_YYYY/HH_MM_SS.log`) và xuất ra 4 luồng ghi cách ly:
  - `oper_logs`: Lưu toàn bộ tiến trình vận hành hệ thống từ mức `INFO` trở lên.
  - `warm_logs`: Ghi nhận các cảnh báo hiệu năng hoặc sự cố từ mức `WARNING` trở lên.
  - `error_logs`: Cách ly các lỗi nghiêm trọng ở mức `ERROR` và `CRITICAL`.
  - `Console (sys.stdout)`: In nhật ký định dạng chuẩn trực tiếp ra màn hình Terminal.

- **`@monitor_performance(time_limit=5.0)`**: Decorator đo lường thời gian thực thi của các hàm xử lý dữ liệu. Tự động tính toán thời gian chạy, xuất log `INFO` khi hoàn thành hoặc xuất cảnh báo `WARNING` vào `oper_logs` và `warm_logs` nếu thời gian chạy vượt quá ngưỡng `time_limit` (tính bằng giây).

- **`@dvc_error_handler(max_retries=3, retry_delay=2, fallback_func=None)`**: Decorator xử lý lỗi tập trung cho các tác vụ tương tác với DVC (Map ngoại lệ, Inject Context, và Fallback & Retry):
  - **Thiếu dữ liệu**: Bắt `PathMissingError`, `FileMissingError`, `FileNotFoundError`, ghi log context (`args`, `kwargs`) và chuyển đổi thành `DataNotFoundError`.
  - **Lỗi phiên bản**: Bắt lỗi `RevisionError` của DVC và chuyển đổi thành `VersionNotFoundError`.
  - **Lỗi kết nối / Remote**: Nhận diện các sự cố mạng (`network`, `connection`, `s3`, `gcs`, `timeout`) hoặc `CalledProcessError`. Tự động thử lại (Retry) tối đa `max_retries` lần (nghỉ `retry_delay` giây). Nếu thất bại hoàn toàn, kích hoạt hàm dự phòng cục bộ (`fallback_func`) trước khi ném ngoại lệ `RemoteStorageError`.
  - **Lỗi hệ thống**: Bắt các ngoại lệ DVC không xác định khác và chuyển thành `DataPipelineException`.

- **`@validation_error_handler()`**: Decorator xử lý lỗi tập trung cho quá trình kiểm định dữ liệu (Data Validation với Pandera):
  - Phân loại chính xác 3 cấp độ lỗi validation: `DataValidationError` (lỗi đã parse), `SchemaErrors` (Lazy Validation vi phạm cấu trúc nhiều dòng), và `SchemaError` (Eager Validation thất bại tại cột cụ thể).
  - Tự động ghi chi tiết log sự cố vào `error_logs` trước khi ngắt luồng (re-raise) để đảm bảo pipeline chính (FastAPI / MLflow Pipeline) dừng an toàn.

### B. MLOps Tracking (Giám sát vòng đời học máy)

*Mã nguồn:* [src/nutrition_core/logging/mlflow_tracker.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/logging/mlflow_tracker.py)

- **`MLOpsTracker` (Class)**: Lớp điều phối trung tâm giao tiếp với MLflow server (mặc định qua SQLite `mlflow.db`). Tự động kích hoạt tính năng theo dõi tài nguyên phần cứng (CPU, GPU, RAM) khi khởi tạo.
  - **`start_run(run_name)`**: Bắt đầu một phiên theo dõi (Run) mới trên MLflow.
  - **`log_params_and_metrics(params, metrics, step)`**: Ghi nhận danh sách các cấu hình/tham số (Params) và các chỉ số đo lường (Metrics) của mô hình hoặc độ trôi dữ liệu.
  - **`log_dataset_lineage(df, dataset_name, source_uri)`**: Tạo liên kết nguồn gốc dữ liệu (Data Lineage) cho Pandas DataFrame, cho phép theo dõi chính xác phiên bản dữ liệu nào đã được dùng để huấn luyện.
  - **`log_error_and_tag(error_msg, tags)`**: Đánh dấu Run hiện tại là `FAILED` và đính kèm trực tiếp thông điệp lỗi (error message) cùng các thẻ (tags) phân loại lên giao diện MLflow UI.
  - **`log_artifact_file(local_path, artifact_path)`**: Hỗ trợ tải các file cục bộ (như biểu đồ, báo cáo Evidently HTML, file dữ liệu quarantine) lên kho lưu trữ Artifacts của Run.
  - **`upload_system_logs()`**: Đọc tự động các file log mới nhất từ `oper_logs` và `error_logs` của hệ thống và đẩy lên mục `system_logs` trong MLflow Artifacts.
  - **`end_run(status="FINISHED")`**: Đóng phiên theo dõi, tự động kích hoạt `upload_system_logs()` trước khi thoát và chốt trạng thái cuối cùng của luồng làm việc.

## 2. Data Input/Output (`src/data_io`)

Phân hệ chịu trách nhiệm trích xuất, biến đổi, nạp dữ liệu (ETL) và quản lý phiên bản dữ liệu/metrics.

*File:* [src/data_io/base_extractor.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/data_io/base_extractor.py)

- **`BasePostgresExtractor` *(ABC)*** : Lớp cơ sở (Abstract Base Class) chịu trách nhiệm quản lý kết nối và trích xuất dữ liệu từ PostgreSQL. Hệ thống hóa quá trình đóng gói bản ghi (Record Packaging) và ánh xạ kiểu dữ liệu (Data Type Mapping) một cách an toàn.

- **`_get_connection()`** : Trích xuất kết nối từ Pool cơ sở dữ liệu (`src.postgres.core.connection`). Tự động bắt lỗi và ném ngoại lệ `DatabaseConnectionError` nếu không thể kết nối, giúp pipeline dừng an toàn.

- **`_convert_data_types(df, schema_mapping, flatten_json_cols)`** : Phương thức lõi xử lý kiểu dữ liệu từ PostgreSQL sang định dạng chuẩn của Pandas/NumPy. Bao gồm 3 luồng xử lý:
  1. *Flatten JSON/JSONB*: Tự động phẳng hóa (flatten) các trường JSON lồng nhau thành các cột độc lập (đặt tên theo định dạng `cột_subcol`).
  2. *Auto Casting*: Xử lý tự động các kiểu dữ liệu thô từ psycopg3/SQLAlchemy:
     - `decimal.Decimal` $\rightarrow$ `float64`
     - `bool` $\rightarrow$ Nullable Boolean (`pd.BooleanDtype`)
     - `int` $\rightarrow$ Nullable Integer (`Int64`)
     - PostgreSQL Array/List $\rightarrow$ `numpy.array`
  3. *Explicit Mapping*: Ép kiểu tường minh dựa trên cấu hình `schema_mapping` truyền vào (hỗ trợ ép kiểu an toàn cho các cột `datetime64`/`datetime64[ns]` thông qua `pd.to_datetime`).

- **`fetch_as_dataframe(query, params, schema_mapping, flatten_json_cols, index_col, mlflow_tracker, dataset_name)`** :
  - Tích hợp decorator `@monitor_performance(time_limit=10.0)` để giám sát thời gian thực thi.
  - Sử dụng `sqlalchemy.text()` kết hợp với `pd.read_sql()` để thực thi truy vấn trực tiếp và đọc kết quả thành DataFrame.
  - Tự động gọi `_convert_data_types` để chuẩn hóa kiểu dữ liệu và thiết lập Index nếu `index_col` được cung cấp.
  - **MLOps Tracking**: Tự động ghi nhận Data Lineage lên MLflow nếu truyền `mlflow_tracker`. Sử dụng định danh nguồn `source_uri = f"nutrition_ai_db_query_{query_hash}"` (loại bỏ tiền tố `postgresql://` để tránh kích hoạt URI Resolver của MLflow).
  - Bắt lỗi thực thi SQL và ném ngoại lệ `QueryExecutionError`.

- **`extract(*args, **kwargs)` *(Abstract Method)*** : Phương thức trừu tượng buộc các lớp con phải triển khai logic trích xuất dữ liệu cụ thể.

*File:* [src/data_io/base_trans.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/data_io/base_trans.py)

- **`BaseTransformer` (ABC)**: Lớp cơ sở trừu tượng cho tất cả các tác vụ Data Transformation (làm sạch, chuẩn hóa, và bảo mật dữ liệu). Tích hợp sẵn `MLOpsTracker` để tự động ghi nhận các chỉ số hiệu năng và lỗi.
  - **`fit(X, y)` / `transform(df)`**: Giao diện tiêu chuẩn hóa (tương tự scikit-learn) yêu cầu các lớp con (subclasses) phải triển khai logic biến đổi dữ liệu cụ thể cho từng pipeline.
  - **`fit_transform(df, y)`**: Thực thi tuần tự luồng `fit` và `transform`. Được tích hợp sẵn decorator `@monitor_performance(time_limit=10.0)`. Tự động tính toán thời gian chạy, đẩy metric `{step_name}_exec_time_seconds` lên MLflow, đồng thời tự động bắt/ghi log khi xảy ra ngoại lệ `DataTransformationError`.
  - **`normalize_column_names(df)`**: Tiện ích chuẩn hóa toàn bộ tên cột của DataFrame. Xử lý an toàn các ký tự Unicode, loại bỏ dấu, và chuyển đổi chuỗi sang định dạng tiêu chuẩn `snake_case` (chữ thường, cách nhau bằng gạch dưới) bằng Regex.
  - **`drop_duplicates_by_key(df, subset, sort_col, keep)`**: Loại bỏ các bản ghi trùng lặp theo khóa. Tự động đếm số lượng bản ghi bị xóa và đẩy chỉ số `{class_name}_dropped_duplicates` lên MLflow, giúp kỹ sư dữ liệu theo dõi tỷ lệ hao hụt dữ liệu qua từng bước.
  - **`hash_pii_columns(df, columns)`**: Tiện ích bảo mật dữ liệu. Chuyển đổi một chiều (băm) các cột chứa thông tin nhạy cảm/định danh cá nhân (PII) bằng thuật toán mã hóa `SHA-256`, đảm bảo tuân thủ quy định bảo mật trước khi lưu trữ hoặc đưa vào huấn luyện mô hình.

*File:* [src/data_io/bulk_loader.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/data_io/bulk_loader.py)

- **`PostgresBulkLoader`**: Lớp chuyên trách xử lý tác vụ nạp dữ liệu khối lượng lớn (Bulk Ingestion) vào PostgreSQL. Được thiết kế để tối ưu hóa tài nguyên bằng cách nhận trực tiếp `connection` từ Connection Pool thay vì liên tục mở kết nối mới.
  - **`execute_bulk_load(table_name, file_path_on_server, schema_name="raw", delimiter, format_type, has_header)`**: Phương thức cốt lõi thực thi luồng tải dữ liệu.
    - **Cơ chế COPY Tốc độ cao**: Sử dụng câu lệnh `COPY` nguyên thủy của PostgreSQL để nạp trực tiếp file (hỗ trợ định dạng CSV/TEXT) từ server vào database, cho tốc độ vượt trội so với lệnh `INSERT` thông thường. Hỗ trợ ánh xạ linh hoạt vào các schema khác nhau (mặc định là `raw`).
    - **Quản lý Transaction (ACID)**: Trích xuất an toàn Raw DBAPI connection từ SQLAlchemy để khởi tạo cursor. Tự động kiểm soát luồng Commit trực tiếp (`raw_conn.commit()`) và xử lý đóng kết nối an toàn trong khối `finally`, đảm bảo không bị treo trỏ (dangling cursors) khi pipeline gặp lỗi.
    - **Giám sát & Tracking tự động**: Được bao bọc bởi decorator `@monitor_performance(time_limit=10.0)`. Tự động đo lường thời gian chạy (ETL execution time), đếm số lượng bản ghi đã nạp (`row_count`), xuất cảnh báo nếu quá trình nạp vượt ngưỡng 10 giây và đồng bộ nhật ký lên MLflow.

*File:* [src/data_io/dvc_handler.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/data_io/dvc_handler.py)

- **Pydantic Schemas (Interface Adaptation)**: Bộ schema định nghĩa cấu trúc chuẩn hóa (Type Hinting & Validation) cho dữ liệu chỉ số đánh giá và siêu dữ liệu của DVC, đảm bảo tính nhất quán trước khi lưu trữ hoặc ghi vào MLflow:
  - `PerClassMetric`: Quản lý các chỉ số chi tiết cho từng nhãn/class phân loại (`precision`, `recall`, `f1_score`, `support`).
  - `GlobalSummaryMetrics`: Quản lý các chỉ số tổng quan của mô hình (`accuracy`, `f1_score`, `precision`, `recall`, `loss`, `roc_auc`).
  - `EpochMetrics`: Lưu vết các thông số theo từng epoch huấn luyện (`best_epoch`, `total_epochs`, `train_loss_final`, `val_loss_final`).
  - `ExecutionMetadata`: Lưu trữ siêu dữ liệu phiên chạy (`dataset_version`, `execution_time_seconds`, `num_samples_eval`, `framework`).
  - `DVCMetricsNormalizedSchema`: Schema wrapper gom cụm toàn bộ các sub-schema trên thành một cấu trúc JSON/Dict thống nhất.

- **`DVCHandler` *(Class)***: Lớp Adapter che giấu sự phức tạp khi tương tác với DVC (kết hợp cả DVC Python API và CLI). Toàn bộ phương thức nghiệp vụ đều được tích hợp Decorator `@dvc_error_handler` để tự động hóa việc bắt lỗi, thử lại (Retry) và kích hoạt dự phòng (Fallback):
  - `__init__(repo=None, rev=None)`: Khởi tạo handler với thông tin repository và phiên bản (revision/git commit) tùy chọn.
  - `_fallback_local_dataset(data_path, format_type)`: Cơ chế dự phòng (Fallback) đọc file trực tiếp từ cache cục bộ (`data/raw/`) nếu việc truy cập DVC Remote thất bại.
  - `sync_data()`: Kéo dữ liệu đồng bộ từ remote storage về máy cục bộ thông qua lệnh `dvc pull` (tự động thử lại tối đa 3 lần).
  - `is_data_updated()`: Kiểm tra trạng thái cập nhật dữ liệu của dự án thông qua câu lệnh `dvc status -c`.
  - `get_dataset(data_path, format_type='csv')`: Stream tệp dữ liệu (`.csv` hoặc `.parquet`) trực tiếp từ DVC Remote thành Pandas DataFrame qua `dvc.api.open`. Đã kích hoạt hàm dự phòng `_fallback_local_dataset` nếu gặp sự cố mạng.
  - `get_latest_model(model_path, framework='sklearn')`: Stream và giải mã mô hình từ DVC Remote thành đối tượng Python, hỗ trợ các framework `scikit-learn` (qua `joblib`) và `PyTorch` (qua `torch`).
  - `get_metrics(metrics_path='metrics.json')`: Đọc file `metrics.json` từ DVC và tự động parse/validate dữ liệu qua `DVCMetricsNormalizedSchema`.
  - `get_image_base64(image_path)`: Đọc dữ liệu ảnh nhị phân từ DVC Remote và mã hóa thành chuỗi Base64 (dùng cho hiển thị UI hoặc tạo báo cáo).
  - `add_data(file_path)`: Đưa file hoặc thư mục mới vào hệ thống quản lý phiên bản của DVC bằng lệnh `dvc add`.

## 3. Quản Lý Ngoại Lệ (Exception Handling)

Phân hệ này cung cấp các lớp ngoại lệ tùy chỉnh và công cụ xử lý lỗi để chuẩn hóa việc bắt lỗi, ghi log và phản hồi trong toàn bộ hệ thống [NutritionAI_V1](https://github.com/KingCarl2000/NutritionAI_V1).

### A. Ngoại lệ Cốt lõi Hệ thống (System Core Exceptions)

*Mã nguồn:* [`src/nutrition_core/exception/exception.py`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/exception/exception.py)

* **`NutritionBaseException`**: Lớp cơ sở (Base class) chứa lỗi hệ thống. Tự động trích xuất thông tin traceback (tên file `file_name` và số dòng `lineno`) thông qua phương thức `_get_detailed_error_info` để xuất thông báo lỗi chính xác.
* **Data Pipeline Exceptions**:
  * **`DataPipelineException`**: Lỗi chung cho quá trình luân chuyển và nạp dữ liệu.
  * **`DataTransformationError`**: Dùng khi có lỗi phát sinh trong quá trình biến đổi dữ liệu (VD: vô tình loại bỏ đặc trưng cần thiết).
  * **`DataCleansingError`**: Dùng khi có lỗi làm sạch dữ liệu (VD: thay thế null sai mã categorical columns).
  * **`DataNotFoundError`**: Lỗi khi không tìm thấy dữ liệu/mô hình (Map từ `dvc.exceptions.PathMissingError`).
  * **`RemoteStorageError`**: Lỗi khi tương tác với các hệ thống lưu trữ từ xa.
* **ML Pipeline Exceptions**:
  * **`MLPipelineException`**: Lỗi chung cho các ML Pipelines.
  * **`ModelTrainingError`**: Lỗi phát sinh trong quá trình huấn luyện (VD: sai tham số khi thiết lập K-means, DBSCAN

## 4. PostgreSQL Database & Feature Store (`src/postgres`)

Phân hệ này quản lý các tác vụ liên quan đến cơ sở dữ liệu PostgreSQL (lưu trữ chính và vector store với pgvector), bao gồm cấu hình, kết nối, luồng nạp dữ liệu cho Feature Store và giám sát hiệu suất truy vấn của [NutritionAI_V1](https://github.com/KingCarl2000/NutritionAI_V1).

### A. Core Database (Cấu hình & Kết nối)

*Mã nguồn:*

- [`src/postgres/core/db_config.py`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/postgres/core/db_config.py)
- [`src/postgres/core/connection.py`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/postgres/core/connection.py)

* **`db_config.py`**:
  * **Tự động định vị môi trường**: Sử dụng `pathlib.Path` để tự động xác định đường dẫn gốc của dự án (`PROJECT_ROOT`) và nạp file cấu hình `.env` thông qua thư viện `dotenv`.
  * **`DatabaseConfig` (Dataclass)**: Lớp lưu trữ cấu hình kết nối PostgreSQL. Tự động đọc và gán các giá trị từ biến môi trường (như `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, v.v.) với các giá trị mặc định an toàn. Lớp này cung cấp hàm `get_sqlalchemy_url()` để xuất chuỗi kết nối đạt chuẩn.
* **`connection.py`**:
  * Tích hợp custom logger (`src.nutrition_core.logging.logger`) thay vì thư viện logging mặc định để đồng bộ chuẩn ghi nhật ký của hệ thống.
  * **`get_engine()`**: Hàm khởi tạo và trả về đối tượng SQLAlchemy Engine kết hợp thư viện `psycopg` mới nhất.
    * **Mẫu thiết kế Singleton**: Sử dụng biến global `_db_engine` để đảm bảo hệ thống chỉ duy trì một Engine duy nhất trong suốt vòng đời chạy ứng dụng, tránh rò rỉ tài nguyên bộ nhớ.
    * **Connection Pooling (Hồ bơi kết nối)**: Tối ưu hóa hiệu suất chịu tải với cấu hình an toàn chặt chẽ: `pool_size=5` (số kết nối thường trực), `max_overflow=10` (số kết nối tối đa mở rộng), `pool_timeout=30`, và `pool_recycle=1800` (làm mới kết nối sau mỗi 30 phút để tránh lỗi rớt mạng ngầm).

### B. Feature Store Engine (Quản lý luồng nạp đặc trưng)

*Mã nguồn:* [`src/postgres/feature_store/engine.py`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/postgres/feature_store/engine.py)

* **`FeatureStoreEngine`**: Lớp điều khiển chính (Engine) chịu trách nhiệm tương tác và nạp dữ liệu vào Feature Store. Được khởi tạo thông qua chuỗi kết nối cơ sở dữ liệu `conn_info`.
  * **`copy_from_csv(table_name, file_path)`**: Phương thức tối ưu hóa để nạp dữ liệu số lượng lớn từ file thô phía máy khách. Đọc trực tiếp từ file CSV và đẩy lên máy chủ PostgreSQL bằng lệnh `COPY FROM STDIN WITH (FORMAT CSV, HEADER true)`. Cách tiếp cận này giúp tránh nghẽn I/O hệ thống khi khởi tạo Feature Store lần đầu.
  * **`stream_features_from_memory(table_name, columns, data_stream)`**: Phương thức nâng cao cho phép đẩy trực tiếp dữ liệu đang nằm trên RAM (ví dụ: kết quả từ Pandas DataFrame trong pipeline MLOps đã được chuyển thành Generator/Iterator chứa các `Tuple`) vào Database. Sử dụng giao thức `COPY` thay vì lệnh `INSERT` từng dòng, giúp tối ưu hóa bộ nhớ và tăng tốc độ luân chuyển dữ liệu động.

### C. Monitoring (Giám sát truy vấn)

*Mã nguồn:* [src/postgres/monitoring/query_analyzer.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/postgres/monitoring/query_analyzer.py)

* **`analyze_query_performance(query: str)`**: Công cụ phân tích và đo lường hiệu năng của các câu lệnh SQL truy vấn vào Feature Store.
  * **Cơ chế hoạt động**: Tự động chèn tiền tố `EXPLAIN (ANALYZE, FORMAT JSON)` vào truy vấn nguyên bản để phân tích cấu trúc thực thi.
  * **Thực thi an toàn**: Quản lý truy vấn an toàn thông qua việc ép kiểu với `psycopg.sql.SQL` và `typing.LiteralString`, sử dụng luồng kết nối trực tiếp từ `get_connection()`.
  * **Giám sát hiệu suất**: Trích xuất kết quả phân tích (execution plan) dưới dạng JSON để lấy thông tin chi tiết về thời gian thực thi thực tế, đồng thời hỗ trợ phát hiện các tình huống rủi ro suy giảm hiệu suất của Database như quét toàn bảng (Sequential Scan / Seq Scan).
  * **Cảnh báo tự động**: Tích hợp custom logger để xuất thông báo (mức `warning`) nếu không thể trích xuất dữ liệu execution plan hợp lệ từ PostgreSQL.

## 5. ML Pipelines

### A. Calories Prediction ( src/pipelines/ml_pipelines/calories_pred )

Phân hệ này quản lý vòng đời dữ liệu đầu vào, kiểm định chất lượng và giám sát mô hình cho dự án NutritionAI_V1.

*File:* [data_ingest.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/pipelines/ml_pipelines/calories_pred/data_ingest.py)

- **`CaloriesDataIngestion(BasePostgresExtractor)` *(Class)***: Lớp điều phối toàn bộ tiến trình trích xuất, tiền xử lý cơ bản, phân chia dữ liệu và xuất dữ liệu ra Feature Store cho pipeline dự đoán lượng calo (`calories_pred`). Kế thừa từ `BasePostgresExtractor`.

- **`__init__(config: dict, tracker: MLOpsTracker = None)`**:
  - Khởi tạo kết nối cơ sở dữ liệu và lớp xử lý DVC (`DVCHandler`).
  - Nhận đối tượng `MLOpsTracker` từ orchestrator để đồng bộ `run_id`, hoặc tự khởi tạo tracker mới với `experiment_name="calories_pred"`.
  - Tải và gán cấu hình schema từ file YAML thông qua `_load_schema()`.

- **`_load_schema()`**:
  - Đọc file cấu hình Dataset Schema định dạng YAML (`yaml.safe_load`) từ đường dẫn `config['schema']['data_config_file']`. Bắt lỗi `FileNotFoundError` và `YAMLError` để ghi log chi tiết.

- **`extract_data_from_postgresql()`**:
  - Decorator: `@monitor_performance(time_limit=30)`
  - Xác định `schema_name` và `table_name` từ `config` hoặc fallback về `dataset_schema`.
  - Thực thi câu lệnh SQL trích xuất dữ liệu thông qua phương thức `fetch_as_dataframe` của base class:
    - Tự động ép kiểu dữ liệu (array, decimal, nullable integers).
    - Đặt cột `User_ID` làm Index.
    - Đồng bộ Data Lineage lên MLflow.

- **`process_and_split_data(df: pd.DataFrame)`**:
  - Decorator: `@monitor_performance()`
  - Loại bỏ các cột không cần thiết dựa trên danh sách `columns_to_drop` trong cấu hình schema.
  - Phân chia dữ liệu thành 3 tập độc lập (Train, Validation, Test) theo tỷ lệ cấu hình (`train_ratio`, `validation_ratio`, `test_ratio`) bằng `train_test_split` với `random_state=42`.

- **`export_to_feature_store(df, train_df, val_df, test_df)`**:
  - Decorator: `@monitor_performance(time_limit=15.0)`
  - Xuất 4 tệp dữ liệu CSV (`current_data.csv`, `train.csv`, `val.csv`, `test.csv`) kèm index vào thư mục Feature Store (`export_dir`, mặc định `Artifacts/features/calories`).
  - Đưa toàn bộ thư mục Feature Store vào quản lý phiên bản của DVC (`self.dvc.add_data(export_dir)`) nếu cấu hình `dvc_tracked=True`.

- **`export_ingestion_report(report_dir, metrics, params)`**:
  - Tạo và lưu báo cáo ingestion định dạng JSON (`ingestion_report.json`) chứa các thông số: tên pipeline, thời gian thực thi (timestamp ISO format), tham số đầu vào, chỉ số số lượng bản ghi và trạng thái execution.
  - Tự động tải file báo cáo này lên MLflow Artifact Store nếu `tracker` đang hoạt động.

- **`extract()`**:
  - Triển khai phương thức trừu tượng (abstract method) từ `BasePostgresExtractor`, gọi trực tiếp `extract_data_from_postgresql()`.

- **`run(report_output_dir: str = None)`**:
  - Phương thức chính vận hành toàn bộ luồng Ingestion trong một MLflow Run (`run_name="data_ingestion"`).
  - Thu thập chỉ số (`total_rows`, `total_features`, `train_rows`, `validation_rows`, `test_rows`) và tham số phân tách dữ liệu để đẩy lên MLflow Tracker.
  - Gọi `export_ingestion_report` xuất báo cáo offline nếu truyền tham số `report_output_dir`.

*File:* [data_val.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/pipelines/ml_pipelines/calories_pred/data_val.py)

* **`load_class_from_registry(api_path: str)`**: Hàm tiện ích giúp nạp động (dynamic import) các lớp kiểm định (schema classes) dựa trên đường dẫn cấu hình trong file YAML. Cơ chế này giúp loại bỏ việc hard-code các import trực tiếp, tăng tính linh hoạt khi thay đổi rule kiểm định.
* **`run_validation_pipeline(...)`**: Hàm lõi thực thi quy trình kiểm định MLOps sử dụng Pandera (với cấu hình `lazy=True` để quét toàn bộ lỗi thay vì dừng ở lỗi đầu tiên).
  * **Cơ chế Quarantine & Rescue (Cách ly và Cứu vãn)**: Tự động tách các bản ghi vi phạm logic (failure_cases) lưu vào thư mục `quarantine_dir` cùng file báo cáo chi tiết. Phần dữ liệu hợp lệ (valid_df) vẫn được "cứu" và lưu trữ dưới dạng `.parquet` cho bước Transformation tiếp theo.
  * **Alerting & Thresholds**: Tính toán tỷ lệ lỗi (`error_rate`). Nếu tỷ lệ này vượt qua ngưỡng cho phép (`error_threshold`), hệ thống lập tức ghi log Error, phát cảnh báo và ngắt Pipeline bằng Exception.
  * **Tích hợp MLflow**: Tự động theo dõi các metrics liên quan đến chất lượng dữ liệu như `val_{mode}_error_rate` và `val_{mode}_invalid_count` (với mode là train, val, hoặc test) vào MLOpsTracker.
* **`CaloriesDataValidationPipeline`**: Lớp điều phối (Orchestrator) tổng thể. Chịu trách nhiệm nạp cấu hình từ `data_val.yaml`, khởi tạo các thư mục (input, output, quarantine), khởi chạy `MLOpsTracker` và duyệt qua toàn bộ các file `.parquet` thô để tiến hành kiểm định hàng loạt.

*File:* [eval_drift.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/pipelines/ml_pipelines/calories_pred/eval_drift.py)

* **`load_yaml_config(path: str)`**: Hàm trợ giúp đọc file cấu hình YAML, tích hợp bắt lỗi và ghi log khi file không tồn tại.
* **`CaloriesDriftEvaluator`**: Lớp cốt lõi đánh giá độ trôi lệch dữ liệu (Data Drift), thiết kế đọc tham số động từ các file cấu hình `evidently_profile.yaml` và `global.yaml`.
  * **Khởi tạo (`__init__`)**: Tự động trích xuất các đường dẫn I/O (dữ liệu tham chiếu, thư mục xuất report), các ngưỡng cảnh báo MLOps (`psi_warning`, `psi_critical`, `p_value_alpha`) và ánh xạ cột (`column_mapping`).
  * **`evaluate_drift(current_df, reference_df)`**: Phương thức đo lường chính, được giám sát hiệu năng với decorator `@monitor_performance(time_limit=10.0)`. Sử dụng `EvidentlyDriftEngine` để xuất báo cáo HTML và JSON. Trích xuất tỷ lệ trôi lệch (`drift_share`) và đối chiếu với các Thresholds để phát ra cảnh báo:
    * **CRITICAL ALERT**: Vượt ngưỡng `psi_critical` (yêu cầu Retrain Model).
    * **WARNING**: Vượt ngưỡng `psi_warning` (cần theo dõi).
    * **INFO**: Dữ liệu ổn định.
* **`run_drift_detection(...)`**: Hàm Wrapper (đóng gói) tiện ích. Cho phép Data Validation Pipeline hoặc Orchestrator gọi thẳng vào quy trình đánh giá Drift một cách gọn gàng, ẩn đi các chi tiết khởi tạo class và cấu hình bên dưới.

## 6. Data Validation (Hệ sinh thái Pandera & evidently AI)

### A. API

*Mã nguồn:* [src/nutrition_core/pandera/api/base.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/pandera/api/base.py)

- *BaseSchema* *(Class)* : Lớp Schema cơ sở kế thừa từ `pa.DataFrameModel`, áp dụng các quy tắc kiểm định khung cho tất cả mô hình dữ liệu (như `CaloriesSchema`, `UserSchema`,...).
    - *Config* : Định nghĩa các quy tắc kiểm định cốt lõi:
        - `coerce = True` : Tự động ép kiểu dữ liệu về đúng dtype khai báo.
        - `strict = False` : Linh hoạt chấp nhận các cột phụ ở tầng Base.
        - `ordered = False` : Không bắt buộc thứ tự cột trừ khi pipeline cụ thể yêu cầu.
        - `unique_column_names = True` : Đảm bảo tên các cột trong DataFrame không bị trùng lặp.
        - `add_missing_columns = True` : Tự động tạo cột và điền default value nếu dữ liệu thiếu cột không bắt buộc.
- *validate_df(cls, df, lazy=True, drop_invalid_rows=False)* *(Classmethod)* : Phương thức thực thi kiểm định DataFrame.
    - Được tích hợp sẵn decorator `@validation_error_handler()` để tự động bắt lỗi `DataValidationError`/`SchemaError` và ghi vào system logs.
    - Tách biệt logic xử lý bằng cách chuyển đổi model sang `DataFrameSchema` và gọi `NutritionPandasBackend` để thực thi.
    - *MLOps Tracking* : Tích hợp tự động với `MLOpsTracker` (pipeline `Data_Validation_Pipeline`). Tự động đẩy kết quả, trạng thái (`SUCCESS`, `PARTIAL_SUCCESS`, `FAILED`) và các metric (như `error_count`, `dropped_rows`) lên MLflow. Tự động xuất JSON danh sách lỗi vào mục Artifacts.
    - *Xử lý lỗi linh hoạt* : Nếu `drop_invalid_rows=True`, hệ thống tự động loại bỏ dòng vi phạm, ghi cảnh báo và trả về DataFrame sạch. Nếu `False`, hệ thống ngắt an toàn luồng pipeline và ném ngoại lệ `DataValidationError`.

*Mã nguồn:* [src/nutrition_core/pandera/api/check.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/pandera/api/check.py)

Module cung cấp các Custom Check Generators & Extensions tổng quát cho hệ thống Pandera. Đây là tập hợp các công cụ (Utilities) độc lập với dataset, cho phép các Pipeline Schema tùy chỉnh truyền tham số và tái sử dụng linh hoạt.

- **1. Extensions API (Phương thức Vectorized dùng chung)**:
    - Được đăng ký qua decorator `@pa_ext.register_check_method`, giúp gọi trực tiếp dưới dạng string trong Schema.
    - `is_in_numeric_range(min_val=0.0, max_val=1e9)` : Kiểm tra giá trị của một cột số có nằm an toàn trong khoảng `[min_val, max_val]` hay không.
    - `is_valid_string_length(min_len=1, max_len=255)` : Ràng buộc độ dài của chuỗi ký tự phải nằm trong khoảng `[min_len, max_len]`.

- **2. GenericCheckFactory** *(Class)*: Tập hợp các Factory Functions giúp khởi tạo linh hoạt các quy tắc kiểm định phức tạp đa cột.
    - `check_sum_le_total(component_cols, total_col, error_msg)` : *[Wide Check]* Đảm bảo tổng giá trị của các cột thành phần (`component_cols`) không vượt quá giá trị của cột tổng (`total_col`) trên cùng một dòng.
    - `check_range_validity(min_col, target_col, max_col, error_msg)` : *[Wide Check]* Đảm bảo logic thứ tự độ lớn hợp lệ: `min_col <= target_col <= max_col`.
    - `check_group_mean_comparison(groupby_col, target_col, lower_group, higher_group, error_msg)` : *[Groupby Check]* Đối chiếu trung bình nhóm, kiểm tra trung bình của cột `target_col` thuộc `lower_group` có nhỏ hơn `higher_group` hay không.
    - `check_regex_pattern(pattern, error_msg)` : *[Element-wise Check]* Xác thực định dạng từng phần tử (chuỗi) dựa trên biểu thức chính quy (Regex).
    - `create_outlier_warning(upper_threshold, warning_msg)` : *[Soft Check]* Chỉ sinh cảnh báo dữ liệu (`SchemaWarning`) khi phát hiện giá trị bất thường vượt ngưỡng cực đại (Outlier), cho phép ghi log mà **không ngắt** luồng thực thi hệ thống.

### B. Pandas Execution Backend (Thực thi kiểm định hệ thống)

*Mã nguồn:* [src/nutrition_core/pandera/backends/pandas_executor.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/pandera/backends/pandas_executor.py)

Module thực thi Backend kiểm định dành riêng cho dữ liệu Pandas DataFrame trong NutritionAI_V1. Đóng vai trò là lớp điều phối trung gian (Facade) giữa Schema API và lõi xử lý, chịu trách nhiệm phân giải Schema, tiền xử lý, thực thi kiểm định vector hóa và gom nhóm lỗi.

- **NutritionPandasBackend** *(Class)* : Backend xử lý và điều phối quá trình kiểm định trên Pandas DataFrame.
    - `resolve_backend(data)` *(Classmethod)* : Xác thực nghiêm ngặt kiểu dữ liệu đầu vào. Đảm bảo dữ liệu phải là `pandas.DataFrame`, ngược lại ném ngoại lệ `TypeError`.
    - `_preprocess_and_coerce(df)` : Tiền xử lý dữ liệu trước khi kiểm định. Hỗ trợ đối chiếu cột bằng Regex, kiểm tra tính toàn vẹn của các cột bắt buộc, và tự động ép kiểu an toàn (coerce) thông qua `pandas_engine`.
    - `_execute_vectorized_checks(df)` : Thực thi lõi kiểm định toàn cục (validate). Gọi trực tiếp `PanderaErrorHandler` để phân tích (parse) các lỗi cấu trúc (Lazy errors) thành danh sách chi tiết, đồng thời trả về bộ lọc (`valid_mask`) định vị chính xác index của các dòng vi phạm.
    - `_record_error(column, reason, index)` : Ghi nhận và lưu trữ thông tin của các ca kiểm định đơn lẻ thất bại.
    - `run_pipeline(df)` : Trái tim của Backend, thực thi Pipeline tuần tự: Tiền xử lý -> Kiểm định Vector hóa -> Xử lý dữ liệu lỗi. Tự động lọc bỏ các dòng vi phạm nếu `drop_invalid_rows=True`. Nếu cấu hình `lazy=False` (Eager execution), hệ thống sẽ ngắt ngay luồng chạy và ném ngoại lệ `DataValidationError` ngay ở lỗi đầu tiên.
- **validate_with_pandas_backend(schema, df, **kwargs)** *(Function)* : API Wrapper tiện ích giúp các module ngoại vi gọi độc lập luồng kiểm định bằng Backend này mà không cần khởi tạo instance. Tự động đánh giá kết quả và ném `DataValidationError` nếu quá trình xử lý tồn đọng lỗi mà không có chỉ định loại bỏ dòng.

### C. Engines



*Mã nguồn:* [src/nutrition_core/pandera/config.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/pandera/config.py)

Module cung cấp cấu hình toàn cục (Global Configuration) cho quá trình kiểm định dữ liệu Pandera trong hệ thống NutritionAI_V1. Chịu trách nhiệm quản lý các mức độ kiểm định dựa trên môi trường triển khai thực tế và tích hợp luồng ghi nhận MLOps.

- **setup_pandera_validation(mlflow_tracker=None)** *(Function)* : Khởi tạo và thiết lập mức độ kiểm định toàn cục (ValidationDepth) tại điểm khởi chạy ứng dụng.
    - *Quản lý môi trường* : Đọc cấu hình từ biến môi trường `PANDERA_VALIDATION_DEPTH` (mặc định là `SCHEMA_AND_DATA`). Ánh xạ tự động sang các Enum chuẩn của Pandera: `SCHEMA_ONLY`, `DATA_ONLY`, hoặc `SCHEMA_AND_DATA`.
    - *Theo dõi hiệu năng* : Tích hợp sẵn decorator `@monitor_performance(time_limit=2.0)` để đảm bảo quá trình tải cấu hình và context diễn ra nhanh chóng, xuất cảnh báo nếu vượt ngưỡng 2 giây.
    - *MLOps Tracking* : Tự động tích hợp với `MLOpsTracker`. Nếu truyền vào một instance tracker hợp lệ, hàm sẽ tự động đẩy tham số cấu hình `pandera_validation_depth` lên MLflow để phục vụ việc tra cứu (Data Lineage & Reproducibility).
    - *Cấu hình luồng* : Trả về một đối tượng `config_context` chứa thiết lập `validation_depth`, giúp giới hạn hoặc mở rộng chi phí tính toán của quá trình kiểm định một cách linh hoạt tùy theo môi trường (Dev/Staging/Prod).

*Mã nguồn:* [src/nutrition_core/pandera/decorators.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/pandera/decorators.py)

Module cung cấp các Decorators tùy chỉnh hỗ trợ kiểm định dữ liệu cho các Data Pipelines. Đóng vai trò bọc (wrap) các decorator gốc của Pandera nhằm mục đích chuẩn hóa xử lý ngoại lệ thành `DataValidationError` và tự động ghi nhận (tracking) log/metadata vào MLflow.

- **nutrition_check_types(*args, **kwargs)** *(Decorator)* : 
    - Tự động kiểm tra tính hợp lệ của dữ liệu đầu vào dựa trên Type Annotations (ví dụ: `df: DataFrame[SchemaModel]`). 
    - Bọc decorator `pa.check_types` của Pandera. 
    - *Xử lý ngoại lệ & MLOps* : Khi phát hiện lỗi `SchemaError` hoặc `SchemaErrors`, tự động trích xuất chi tiết lỗi thông qua `PanderaErrorHandler`, ghi nhận thất bại lên hệ thống MLflow (thông qua `MLOpsTracker`), và ném ra `DataValidationError` để dừng pipeline an toàn.
- **nutrition_check_io(head=5, tail=5, **kwargs)** *(Decorator)* : 
    - Kiểm định tường minh (explicit validation) cho Input và Output của hàm thông qua `DataFrameSchema`. Khai báo linh hoạt qua `kwargs` (ví dụ: `df=schema_in`, `out=schema_out`).
    - Bọc decorator `pa.check_io` của Pandera. Cung cấp tham số `head` và `tail` để cấu hình số lượng dòng lỗi mẫu xuất ra log.
    - *Tích hợp an toàn* : Tương tự `nutrition_check_types`, đảm bảo bắt mọi lỗi cấu trúc I/O, đẩy log lên MLflow và chuẩn hóa ngoại lệ ném ra hệ thống.

*Mã nguồn:* [`src/nutrition_core/pandera/evidently_engine.py`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/pandera/evidently_engine.py)

Module cung cấp `EvidentlyDriftEngine` – engine tích hợp khép kín giữa kiểm định lược đồ dữ liệu động (Pandera Schema Validation), phân tích trôi lệch dữ liệu (Evidently AI Data Drift Engine), quản lý phiên bản (DVC Artifact Management) và giám sát MLOps (MLflow Logging) trong hệ thống `NutritionAI_V1`.

- **EvidentlyDriftEngine** *(Class)* : Lớp quản lý và điều phối toàn bộ chuỗi kiểm định dữ liệu và trôi lệch mô hình.
    - `__init__(artifact_dir, project_name, evidently_workspace_url, mlops_tracker)` : Khởi tạo engine với thư mục lưu trữ artifact bắt buộc, tên dự án, đường dẫn workspace của Evidently và instance theo dõi MLOps. Tự động tạo thư mục đích nếu chưa tồn tại.
    - `validate_schema(df, schema_model)` *(Giai đoạn 1)* : Thực thi kiểm tra cấu trúc/lược đồ DataFrame động thông qua Pandera Schema Model. Ghi log trạng thái thành công (`PASSED`) hoặc thất bại (`FAILED`) ra hệ thống.
    - `run_drift_detection(current_df, reference_df, column_mapping_cfg, drift_options_cfg)` *(Giai đoạn 2)* : Phân tích trôi lệch dữ liệu bằng Evidently AI dựa trên file cấu hình YAML. 
        - Trích xuất cấu hình đặc trưng (numerical, categorical, text) qua `DataDefinition`.
        - Xây dựng preset kiểm định (`DataDriftPreset` kết hợp `DataSummaryPreset`) với các ngưỡng linh hoạt (`drift_share`, `p_value_alpha`, `wasserstein_distance`, v.v.).
        - Xuất báo cáo tài nguyên dưới dạng file HTML (`drift_report.html`) và chỉ số định lượng JSON (`drift_metrics.json`) vào thư mục `artifact_dir`.
        - Hỗ trợ đồng bộ tự động Snapshot lên Evidently Remote Workspace UI nếu có cấu hình URL.
    - `log_metrics_to_mlflow(validation_json)` *(Giai đoạn 3)* : Đẩy các chỉ số trôi lệch quan trọng (`drift_share`, `number_of_drifted_columns`, `dataset_drift_detected`) cùng bộ tệp báo cáo Artifacts lên hệ thống MLflow thông qua `MLOpsTracker`.

## 7. Database & CRUD (`src/nutrition_core/database`)

Phân hệ này quản lý việc định nghĩa các cấu trúc bảng (ORM Models) và phiên giao dịch (Sessions) để tương tác với cơ sở dữ liệu. Điểm đặc biệt của kiến trúc này là khả năng sinh Model động từ file cấu hình thay vì hard-code.

### A. Session Management (Quản lý phiên kết nối)

*Mã nguồn:* [src/nutrition_core/database/session.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/database/session.py)

* **Khởi tạo `SessionLocal`**: Khởi tạo nhà máy tạo Session (`sessionmaker`) của SQLAlchemy để cấp phát các phiên làm việc với cơ sở dữ liệu.
* **Đảm bảo tính toàn vẹn (ACID)**: Cấu hình `autocommit=False` và `autoflush=False` giúp hệ thống kiểm soát chặt chẽ thời điểm xác nhận (commit) hoặc hoàn tác (rollback) dữ liệu, tránh tình trạng ghi dữ liệu rác khi các pipeline xảy ra lỗi đột ngột.
* **Tích hợp Engine & Logging**: Tự động kết nối với hàm `get_engine()` (từ `src.postgres.core.connection`) và tích hợp custom logger để theo dõi toàn bộ trạng thái khởi tạo.

### B. Dynamic Modeling (Định nghĩa Model động)

*Mã nguồn:* [src/nutrition_core/database/models.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/database/models.py)

Phân hệ này được thiết kế để tự động hóa hoàn toàn việc tạo cấu trúc bảng ORM dựa trên các định nghĩa khai báo trong file cấu hình YAML.

* **`TYPE_MAPPING`**: Từ điển ánh xạ các chuỗi định nghĩa kiểu dữ liệu trong YAML (như `"integer"`, `"string"`, `"boolean"`, `"date"`, `"datetime"`) sang các lớp kiểu dữ liệu chuẩn tương ứng của SQLAlchemy (`Integer`, `String`, `Boolean`, `Date`, `DateTime`).
* **`build_model_from_yaml(yaml_filename: str)`**: Hàm cốt lõi chịu trách nhiệm sinh Model động.
  * **Cơ chế hoạt động**: Nhận tên file YAML (nằm trong thư mục `config/datasets/`), tự động đọc và phân tích cấu trúc, bao gồm cả việc xử lý các dictionary lồng nhau (nested dictionary) và các thuộc tính cột.
  * **Đăng ký tự động**: Các Model (Lớp) sinh ra sẽ tự động được trả về và đăng ký vào `Base.metadata`, giúp hệ thống ORM (như Alembic) có thể nhận diện ngay lập tức để thực hiện migration.
  * **Bắt lỗi cấu hình**: Nếu không tìm thấy file cấu hình được chỉ định, hàm sẽ ghi log lỗi ở mức ERROR và ném ra ngoại lệ `FileNotFoundError`, ngăn chặn hệ thống chạy với một database rỗng.

## 8. Automation Scripts (`src/scripts`)

Phân hệ này chứa các đoạn mã tự động hóa quy trình vận hành hệ thống, giúp kỹ sư chạy các tác vụ Data Ingestion, khởi tạo Database, hoặc luồng MLOps một cách độc lập và hàng loạt từ Command Line.

### A. Tự động hóa Bulk Load (Nạp dữ liệu lớn)

*Mã nguồn:* [`src/scripts/run_db_bulk_load.py`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/scripts/run_db_bulk_load.py)

* **Định tuyến động (`PROJECT_ROOT`)**: Tự động xác định đường dẫn gốc của dự án bằng `pathlib.Path` và đưa vào `sys.path`, cho phép thực thi script từ bất kỳ thư mục nào mà không bị lỗi module import.
* **YAML-Driven Ingestion (Nạp qua cấu hình)**:
  * Hàm `main(config_rel_path)` đọc trực tiếp các thông số nạp dữ liệu từ file YAML (ví dụ: `config/datasets/calories.yaml`).
  * Hỗ trợ duyệt qua nhiều bảng (`dataset_key`, `dataset_info`) được định nghĩa trong cùng một file cấu hình để thực thi nạp hàng loạt.
* **Tích hợp `PostgresBulkLoader`**: Khởi tạo bộ nạp dữ liệu và gọi hàm `execute_bulk_load` cho từng file `.csv` thô, tận dụng cơ chế `COPY TO STDIN` của PostgreSQL để tối ưu tốc độ nạp thay vì dùng lệnh INSERT.
* **Thiết kế dễ mở rộng**: Khối `if __name__ == "__main__":` duy trì một danh sách `config_files`. Kỹ sư chỉ cần mở comment (ví dụ: `"config/datasets/recipes.yaml"`) để luồng script tự động quét và nạp thêm các Domain dữ liệu mới vào DB.

### B. Tự động hóa Khởi tạo Bảng (Database Schema Creation)

*Mã nguồn:* [src/scripts/create_tables.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/scripts/create_tables.py)

Script này chịu trách nhiệm đọc cấu hình lược đồ dữ liệu và tự động sinh ra các bảng vật lý trong cơ sở dữ liệu PostgreSQL, loại bỏ hoàn toàn việc phải viết các câu lệnh SQL DDL thủ công.

* **Ánh xạ kiểu dữ liệu (`get_pg_data_type`)**: Tự động chuyển đổi các kiểu dữ liệu chung được khai báo trong file YAML (như `string`, `int`, `boolean`, `datetime`) sang các kiểu dữ liệu chuẩn tối ưu của PostgreSQL (như `VARCHAR(255)`, `BIGINT`, `BOOLEAN`, `TIMESTAMP`).
* **Chuẩn hóa định danh (`normalize_identifier`)**: Xử lý và làm sạch tên các tập dữ liệu (dataset) để tạo ra các tiền tố (prefix) bảng an toàn cho PostgreSQL, ngăn chặn các lỗi cú pháp do ký tự đặc biệt hoặc khoảng trắng gây ra.
* **Luồng thực thi lõi (`create_tables_from_yaml`)**:
  * Kết nối an toàn với Database thông qua `get_connection()`.
  * Tự động sinh và thực thi chuỗi lệnh DDL: Đảm bảo Schema tồn tại (`CREATE SCHEMA IF NOT EXISTS`), dọn dẹp bảng cũ (`DROP TABLE IF EXISTS CASCADE`), và khởi tạo bảng mới (`CREATE TABLE`).
  * **Cơ chế Báo cáo (Report Generation)**: Sau khi tạo bảng thành công, script sẽ tự động tổng hợp và xuất ra file báo cáo `tables_created_report.yaml` (lưu tại `docs/architecture/`). File báo cáo này đóng vai trò cực kỳ quan trọng làm "nguồn sự thật" (Source of Truth) đầu vào cho các luồng Ingestion hoặc Automation Scripts khác (như `run_db_bulk_load.py`) biết chính xác bảng nào đã sẵn sàng để nạp dữ liệu.

### C. Tự động hóa Tạo Schema Pipeline (Pipeline Schema Generation)

*Mã nguồn:* [src/scripts/create_schema_pipelines.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/scripts/create_schema_pipelines.py)

Script này đóng vai trò cầu nối dữ liệu (Data Bridge) giữa hệ thống Cơ sở dữ liệu và các luồng Machine Learning (ML Pipelines). Nó đọc báo cáo cấu trúc bảng đã khởi tạo và tự động sinh ra các file cấu hình JSON chuyên biệt (ví dụ: `calories_data_schema.json`) phục vụ cho khâu trích xuất và biến đổi dữ liệu.

* **Kế thừa Nguồn sự thật (Single Source of Truth)**: Tự động đọc file `docs/architecture/tables_created_report.yaml` (được sinh ra từ bước tạo bảng). Việc này đảm bảo luồng ML luôn đồng bộ tuyệt đối với cấu trúc vật lý của PostgreSQL mà không cần kỹ sư phải khai báo lại cấu trúc bằng tay.
* **Phân loại Cột Tự động (Smart Column Categorization)**: Dựa vào thuộc tính `yaml_type`, script tự động phân tách và gom nhóm các đặc trưng thành:
  * `numerical_cols`: Các cột có kiểu `integer` hoặc `float` (dùng cho Standard Scaling, tính toán độ lệch).
  * `categorical_cols`: Các cột có kiểu `categorical` (dùng cho One-Hot Encoding hoặc Entity Embedding).
* **Kiến trúc Đăng ký Mở rộng (`PIPELINES_REGISTRY`)**: Quản lý tập trung các cấu hình pipeline thông qua một danh sách dictionary. Để tích hợp một domain dữ liệu mới (ví dụ: `recipes`), kỹ sư chỉ cần khai báo một block mới trong biến `PIPELINES_REGISTRY` (gồm `source_dataset`, `dir_path`, `index`, `drop_cols`). Script sẽ tự động quét và render file JSON cấu hình tương ứng.
* **Tích hợp đầu ra JSON**: Gói toàn bộ metadata (cột index, các cột cần loại bỏ, phân loại numerical/categorical) thành một file `.json` hoàn chỉnh. File này sau đó sẽ được `BasePostgresExtractor` và các module Data Ingestion nạp vào để chuẩn hóa DataFrame tự động.

### D. Tự động hóa Báo cáo EDA (EDA Report Generation)

*Mã nguồn:* [src/scripts/eda_report.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/scripts/eda_report.py)

Script này tự động hóa quá trình phân tích khám phá dữ liệu (Exploratory Data Analysis - EDA), xuất ra các báo cáo trực quan và số liệu thống kê chi tiết cho từng bảng dữ liệu trong PostgreSQL.

* **Quản lý cấu hình động**: Tự động đọc định nghĩa các bảng cần phân tích từ file cấu hình `schema.yaml` (nằm trong `src/api/data_schema/`).
* **Trích xuất mẫu dữ liệu an toàn**: Tận dụng `get_connection()` từ Connection Pool để truy vấn dữ liệu từ PostgreSQL. Sử dụng giới hạn (`LIMIT 10000`) thông qua SQL để tránh tình trạng quá tải bộ nhớ (OOM) khi phân tích các bảng dữ liệu khổng lồ.
* **Tính toán Metric tùy chỉnh (`calculate_custom_metrics`)**:
  * **Table-level**: Thống kê số dòng, số cột, tỷ lệ dữ liệu thiếu (missing) và dữ liệu trùng lặp (duplicate).
  * **Numerical metrics**: Tính toán chuyên sâu cho dữ liệu số (min, max, mean, median, IQR, std_dev, skewness, kurtosis, zero ratio).
  * **Categorical/Datetime metrics**: Đo lường tần suất (mode, frequency), số lượng giá trị duy nhất (distinct count) và tự động nhận diện ép kiểu thời gian.
  * **Relationships**: Tự động tính toán ma trận tương quan theo phương pháp Pearson và Spearman.
* **Kết xuất báo cáo kép (Dual Output)**: Kết quả phân tích được lưu tự động vào thư mục lưu trữ tĩnh `Artifacts/EDA_reports/` dưới 2 định dạng riêng biệt:
  1. **Cấu trúc JSON (`{table_name}_metrics.json`)**: Lưu trữ các custom metrics dưới dạng JSON để tiện cho việc tái sử dụng bằng máy trong các luồng MLOps tiếp theo.
  2. **Trực quan HTML (`{table_name}_dataset_schema.html`)**: Báo cáo trực quan sinh động, chứa đồ thị tương tác được tạo tự động bởi thư viện `ydata-profiling` (`ProfileReport`).

### E. Tự động hóa Sinh khung kiểm định (Schema Skeleton Generation)

*Mã nguồn:* [src/scripts/generate_data_schema.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/scripts/generate_data_schema.py)

Script này đóng vai trò tự động hóa việc tạo bộ khung (skeleton) cho các file cấu hình Pandera Schema, giúp giảm thiểu tối đa thao tác viết mã thủ công lặp đi lặp lại (boilerplate code) cho Data Engineer.

* **Kế thừa Metrics tĩnh**: Tự động đọc và phân tích cấu trúc file `{table_name}_metrics.json` (được sinh ra từ bước chạy `eda_report.py`) để trích xuất chính xác danh sách các cột dạng số (numerical) và phân loại/thời gian (categorical/datetime).
* **Tự động sinh mã nguồn (Code Generation)**:
  * Dựa vào tên pipeline (`pipeline_name`), script tự động chuẩn hóa tên Class (ví dụ: `ml_calories` thành `MlCaloriesInputSchema`).
  * Render ra một file mã nguồn Python (`.py`) hoàn chỉnh định nghĩa các lớp Schema kế thừa từ `pa.DataFrameModel`.
  * Thiết lập sẵn các cấu hình bảo vệ cốt lõi: `coerce = True`, `strict = True` và đặt mặc định các trường là bắt buộc (`nullable=False`).
* **Phân luồng Target Column**: Tự động nhận diện cột mục tiêu dự đoán (`target_col`) để tách biệt kiến trúc. Cột mục tiêu sẽ chỉ được khai báo trong `TrainingSchema` (kế thừa từ `InputSchema`), giữ cho `InputSchema` thuần túy chỉ chứa các đặc trưng đầu vào (features).
* **Cơ chế Human-in-the-loop (Chờ chuyên gia tinh chỉnh)**:
  * Script chỉ đóng vai trò tạo khung định kiểu cơ bản (`float` cho số, `str` cho chuỗi).
  * Nó tự động chèn các ghi chú `TODO` bên cạnh mỗi thuộc tính (Ví dụ: `TODO: Xác nhận lại kiểu (int/float) và cập nhật ngưỡng` hoặc `TODO: Cập nhật danh sách isin=[...]`). Điều này định hướng trực tiếp cho Data Engineer/Domain Expert vị trí cần can thiệp thủ công để điền các quy tắc logic nghiệp vụ và ngưỡng sinh học chính xác.
* **Thực thi hàng loạt (Batch Execution)**: Khối cấu hình trung tâm `ML_PIPELINES_CONFIG` cho phép ánh xạ 1-1 giữa nhiều file JSON metrics và đường dẫn xuất schema đầu ra, dễ dàng sinh khung cho hàng loạt Domain cùng lúc.

### F. Tự động hóa cấu hình DVC Pipelines (DVC Stages Setup)

*Mã nguồn:* [src/scripts/setup_dvc_stages.sh](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/scripts/setup_dvc_stages.sh)

Script Bash này tự động hóa việc khởi tạo và liên kết các bước (stages) trong luồng máy học thành một pipeline hoàn chỉnh thông qua Data Version Control (DVC). Việc chạy script này sẽ tạo/cập nhật file `dvc.yaml`, đảm bảo tính tái lập (reproducibility) cho toàn bộ quy trình MLOps của phân hệ Calories Prediction.

* **Tự động định vị thư mục (Auto-Navigation)**: Sử dụng lệnh `cd "$(dirname "$0")/../.."` để luôn di chuyển luồng thực thi về thư mục gốc của dự án `NutritionAI_V1`. Điều này cực kỳ quan trọng giúp DVC nhận diện chính xác các đường dẫn tương đối (relative paths) của dữ liệu và mã nguồn, bất kể bạn gọi script từ thư mục nào.
* **Định nghĩa 5 giai đoạn cốt lõi (ML Pipeline Stages)**: Script thiết lập 5 stage nối tiếp nhau, đầu ra (`-o`) của bước trước làm đầu vào (`-d`) cho bước sau:
  1. **`calories_ingest` (Trích xuất dữ liệu)**: Theo dõi mã nguồn `data_ingest.py` và cấu hình YAML. Xuất ra file khung `calories_data_schema.json` và dữ liệu thô tại `Artifacts/data/ml_calories/raw`.
  2. **`calories_validate` (Kiểm định chất lượng)**: Nhận dữ liệu thô từ bước 1. Theo dõi `data_val.py` và `calories_schema.py`. Phân tách đầu ra thành dữ liệu sạch (`validated`) và dữ liệu lỗi (`quarantine`).
  3. **`calories_transform` (Biến đổi dữ liệu)**: Dựa trên báo cáo kiểm định `val_report.json`, chạy `data_trans.py` để xử lý đặc trưng, xuất ra file dữ liệu sẵn sàng huấn luyện `processed_data.parquet`.
  4. **`calories_train` (Huấn luyện mô hình)**: Kích hoạt `Orchestrator.py` cùng với file tham số `train_params.yaml`. Đầu ra được lưu trữ tĩnh là file trọng số mô hình `model.pkl` tại thư mục Artifacts.
  5. **`calories_eval_drift` (Đánh giá trôi dữ liệu)**: Bước cuối cùng dùng `eval_drift.py` và cấu hình Evidently để so sánh dữ liệu mới và cũ, kết xuất báo cáo trôi lệch dưới dạng `drift_metrics.json` và `drift_report.html`.

## 10. API Framework ( src/api )
