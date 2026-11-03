# DEVELOPER GUIDE: NutritionAI_V1 Core Modules

Tài liệu này tổng hợp danh sách các class, hàm và decorator thuộc phân hệ `src`, giúp nhà phát triển dễ dàng tra cứu và tái sử dụng khi mở rộng hệ thống.

## 1. Logging & MLOps Tracking

Phân hệ này chịu trách nhiệm quản lý nhật ký hệ thống (system logs) nhiều cấp độ và tự động hóa việc theo dõi vòng đời dữ liệu/mô hình thông qua MLflow.

### A. System Logging (Ghi nhận nhật ký hệ thống)
*Mã nguồn:* [src/nutrition_core/logging/logger.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/logging/logger.py)

- **`logger` (Instance)**: Đối tượng logging lõi của dự án (`NutritionAI`). Được cấu hình để tự động phân luồng nhật ký theo định dạng thời gian thực và ghi ra 3 thư mục cách ly:
  - `oper_logs`: Lưu toàn bộ tiến trình hệ thống từ mức `INFO` trở lên.
  - `warm_logs`: Lưu các cảnh báo từ mức `WARNING` trở lên.
  - `error_logs`: Cách ly các lỗi nghiêm trọng ở mức `ERROR` và `CRITICAL`.
- **`@monitor_performance(time_limit=5.0)`**: Decorator dùng để bao bọc các hàm xử lý dữ liệu nặng. Tự động đo lường thời gian thực thi và xuất cảnh báo (Warning) vào log nếu vượt quá ngưỡng `time_limit` (tính bằng giây).
- **`@validation_error_handler()`**: Decorator chuyên dụng để bắt và bọc các lỗi ngoại lệ (đặc biệt là `SchemaErrors` sinh ra từ Pandera), chuẩn hóa thông điệp lỗi hệ thống trước khi ngắt tiến trình.

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

- **`BasePostgresExtractor` (ABC)**: Lớp cơ sở (Abstract Base Class) chịu trách nhiệm quản lý kết nối và trích xuất dữ liệu từ PostgreSQL. Hệ thống hóa quá trình đóng gói bản ghi (Record Packaging) và ánh xạ kiểu dữ liệu (Data Type Mapping) một cách an toàn.
  - **`_get_connection()`**: Trích xuất kết nối từ Pool cơ sở dữ liệu (`src.postgres.core.connection`). Tự động bắt lỗi và ném ngoại lệ `DatabaseConnectionError` nếu không thể kết nối, giúp pipeline dừng an toàn.
  - **`_convert_data_types(df, schema_mapping, flatten_json_cols)`**: Phương thức lõi xử lý kiểu dữ liệu từ PostgreSQL sang định dạng chuẩn của Pandas/NumPy. Bao gồm 3 luồng xử lý:
    1. **Flatten JSON/JSONB**: Tự động phẳng hóa (flatten) các trường JSON lồng nhau thành các cột độc lập (đặt tên theo định dạng `cột_subcol`).
    2. **Auto Casting**: Xử lý tự động các kiểu thô từ `psycopg3` (VD: `decimal.Decimal` chuyển thành `float64`, kiểu `bool` sang `pd.BooleanDtype`, và đặc biệt ép kiểu PostgreSQL Array thành `numpy.array` để phục vụ trực tiếp cho pgvector/embeddings).
    3. **Explicit Mapping**: Ép kiểu tường minh dựa trên cấu hình `schema_mapping` truyền vào (hỗ trợ xử lý an toàn các cột `datetime64`).
  - **`fetch_as_dataframe(query, params, index_col, ...)`**: Thực thi truy vấn SQL (`cursor.execute`), lấy dữ liệu (`fetchall`), đóng gói thành DataFrame, tự động gọi hàm chuyển đổi kiểu dữ liệu ở trên và thiết lập cột Index (`set_index`) nếu được yêu cầu. Hỗ trợ ghi nhận Data Lineage trực tiếp nếu được truyền đối tượng MLflow Tracker.

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

- **Pydantic Schemas (Interface Adaptation)**: Bộ schema định nghĩa các cấu trúc chuẩn hóa (Type Hinting & Validation) cho chỉ số đánh giá mô hình, đảm bảo tính nhất quán trước khi lưu trữ hoặc đẩy lên MLflow:
  - `GlobalSummaryMetrics`: Quản lý các chỉ số tổng quan (accuracy, f1_score, precision, recall, loss, roc_auc).
  - `PerClassMetric`: Quản lý chỉ số chi tiết cho từng nhãn/class phân loại.
  - `EpochMetrics`: Lưu vết các thông số theo quá trình huấn luyện (best_epoch, total_epochs, loss final).
  - `ExecutionMetadata`: Lưu trữ siêu dữ liệu phiên chạy (thời gian thực thi, dataset version, framework sử dụng).
  - `DVCMetricsNormalizedSchema`: Lớp schema bao ngoài (wrapper) gom cụm toàn bộ các sub-schema trên thành một cấu trúc JSON/Dict thống nhất.

- **`DVCHandler`**: Lớp Adapter che giấu sự phức tạp khi tương tác với DVC (API/CLI). Được tích hợp Decorator xử lý lỗi tập trung (`@dvc_error_handler`).
  - **`sync_data()`**: Kéo dữ liệu đồng bộ từ remote storage thông qua lệnh `subprocess.run(["dvc", "pull"])`. Tích hợp cơ chế tự động thử lại (Retry: max_retries=3, delay=2s) khi gặp sự cố mạng lưới.
  - **`_fallback_local_dataset(data_path, format_type)`**: Cơ chế dự phòng (Fallback) thông minh. Nếu quá trình đồng bộ DVC thất bại, hệ thống sẽ tự động chuyển hướng tìm và đọc file dữ liệu trực tiếp từ bộ đệm cục bộ (`data/raw/`), đảm bảo pipeline không bị gián đoạn.

## 3. Quản Lý Ngoại Lệ (Exception Handling)

Phân hệ này cung cấp các lớp ngoại lệ tùy chỉnh và công cụ xử lý lỗi để chuẩn hóa việc bắt lỗi, ghi log và phản hồi trong toàn bộ hệ thống [NutritionAI_V1](https://github.com/KingCarl2000/NutritionAI_V1).

### A. Ngoại lệ Cốt lõi Hệ thống (System Core Exceptions)
*Mã nguồn:* [`src/nutrition_core/exception/exception.py`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/exception/exception.py)

*   **`NutritionBaseException`**: Lớp cơ sở (Base class) chứa lỗi hệ thống. Tự động trích xuất thông tin traceback (tên file `file_name` và số dòng `lineno`) thông qua phương thức `_get_detailed_error_info` để xuất thông báo lỗi chính xác.
*   **Data Pipeline Exceptions**:
    *   **`DataPipelineException`**: Lỗi chung cho quá trình luân chuyển và nạp dữ liệu.
    *   **`DataTransformationError`**: Dùng khi có lỗi phát sinh trong quá trình biến đổi dữ liệu (VD: vô tình loại bỏ đặc trưng cần thiết).
    *   **`DataCleansingError`**: Dùng khi có lỗi làm sạch dữ liệu (VD: thay thế null sai mã categorical columns).
    *   **`DataNotFoundError`**: Lỗi khi không tìm thấy dữ liệu/mô hình (Map từ `dvc.exceptions.PathMissingError`).
    *   **`RemoteStorageError`**: Lỗi khi tương tác với các hệ thống lưu trữ từ xa.
*   **ML Pipeline Exceptions**:
    *   **`MLPipelineException`**: Lỗi chung cho các ML Pipelines.
    *   **`ModelTrainingError`**: Lỗi phát sinh trong quá trình huấn luyện (VD: sai tham số khi thiết lập K-means, DBSCAN

## 4. PostgreSQL Database & Feature Store (`src/postgres`)

Phân hệ này quản lý các tác vụ liên quan đến cơ sở dữ liệu PostgreSQL (lưu trữ chính và vector store với pgvector), bao gồm cấu hình, kết nối, luồng nạp dữ liệu cho Feature Store và giám sát hiệu suất truy vấn của [NutritionAI_V1](https://github.com/KingCarl2000/NutritionAI_V1).

### A. Core Database (Cấu hình & Kết nối)
*Mã nguồn:* 
- [`src/postgres/core/db_config.py`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/postgres/core/db_config.py)
- [`src/postgres/core/connection.py`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/postgres/core/connection.py)

*   **`db_config.py`**:
    *   **Tự động định vị môi trường**: Sử dụng `pathlib.Path` để tự động xác định đường dẫn gốc của dự án (`PROJECT_ROOT`) và nạp file cấu hình `.env` thông qua thư viện `dotenv`.
    *   **`DatabaseConfig` (Dataclass)**: Lớp lưu trữ cấu hình kết nối PostgreSQL. Tự động đọc và gán các giá trị từ biến môi trường (như `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, v.v.) với các giá trị mặc định an toàn. Lớp này cung cấp hàm `get_sqlalchemy_url()` để xuất chuỗi kết nối đạt chuẩn.
*   **`connection.py`**:
    *   Tích hợp custom logger (`src.nutrition_core.logging.logger`) thay vì thư viện logging mặc định để đồng bộ chuẩn ghi nhật ký của hệ thống.
    *   **`get_engine()`**: Hàm khởi tạo và trả về đối tượng SQLAlchemy Engine kết hợp thư viện `psycopg` mới nhất.
        *   **Mẫu thiết kế Singleton**: Sử dụng biến global `_db_engine` để đảm bảo hệ thống chỉ duy trì một Engine duy nhất trong suốt vòng đời chạy ứng dụng, tránh rò rỉ tài nguyên bộ nhớ.
        *   **Connection Pooling (Hồ bơi kết nối)**: Tối ưu hóa hiệu suất chịu tải với cấu hình an toàn chặt chẽ: `pool_size=5` (số kết nối thường trực), `max_overflow=10` (số kết nối tối đa mở rộng), `pool_timeout=30`, và `pool_recycle=1800` (làm mới kết nối sau mỗi 30 phút để tránh lỗi rớt mạng ngầm).

### B. Feature Store Engine (Quản lý luồng nạp đặc trưng)
*Mã nguồn:* [`src/postgres/feature_store/engine.py`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/postgres/feature_store/engine.py)

*   **`FeatureStoreEngine`**: Lớp điều khiển chính (Engine) chịu trách nhiệm tương tác và nạp dữ liệu vào Feature Store. Được khởi tạo thông qua chuỗi kết nối cơ sở dữ liệu `conn_info`.
    *   **`copy_from_csv(table_name, file_path)`**: Phương thức tối ưu hóa để nạp dữ liệu số lượng lớn từ file thô phía máy khách. Đọc trực tiếp từ file CSV và đẩy lên máy chủ PostgreSQL bằng lệnh `COPY FROM STDIN WITH (FORMAT CSV, HEADER true)`. Cách tiếp cận này giúp tránh nghẽn I/O hệ thống khi khởi tạo Feature Store lần đầu.
    *   **`stream_features_from_memory(table_name, columns, data_stream)`**: Phương thức nâng cao cho phép đẩy trực tiếp dữ liệu đang nằm trên RAM (ví dụ: kết quả từ Pandas DataFrame trong pipeline MLOps đã được chuyển thành Generator/Iterator chứa các `Tuple`) vào Database. Sử dụng giao thức `COPY` thay vì lệnh `INSERT` từng dòng, giúp tối ưu hóa bộ nhớ và tăng tốc độ luân chuyển dữ liệu động.

### C. Monitoring (Giám sát truy vấn)
*Mã nguồn:* [`src/postgres/monitoring/query_analyzer.py`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/postgres/monitoring/query_analyzer.py)

*   **`analyze_query_performance(query)`**: Công cụ phân tích và đo lường hiệu năng của các truy vấn gửi đến Feature Store. Hàm này tự động chèn tiền tố `EXPLAIN (ANALYZE, FORMAT JSON)` vào câu lệnh SQL nguyên bản, thực thi và trích xuất các thông tin quan trọng như thời gian thực thi thực tế, đồng thời giúp phát hiện các tình huống rủi ro như quét toàn bảng (Sequential Scan) làm chậm hệ thống.

## 5. ML Pipelines

### 1. Calories Prediction ( src/pipelines/ml_pipelines/calories_pred )

Phân hệ này quản lý vòng đời dữ liệu đầu vào, kiểm định chất lượng và giám sát mô hình cho dự án NutritionAI_V1.

*File:* [data_ingest.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/pipelines/ml_pipelines/calories_pred/data_ingest.py)
*   **`CaloriesDataIngestion`**: Lớp chịu trách nhiệm trích xuất, tiền xử lý cơ bản và chia tách dữ liệu cho bài toán phân tích/dự đoán Calories. Lớp này kế thừa từ `BasePostgresExtractor` nhằm kế thừa cơ chế kết nối DB và ép kiểu (auto casting), đồng thời tích hợp trực tiếp `DVCHandler` và `MLOpsTracker`.
    *   **Thiết kế linh hoạt (YAML-Driven Config)**: Chuyển dịch thiết kế từ việc đọc cứng (hard-code) bằng JSON Schema sang đọc cấu hình động (tên DB/Schema, danh sách cột cần bỏ, cấu trúc chia tập) từ file YAML.
    *   **`extract_data_from_postgresql()`**: Lấy tên bảng từ YAML (với fallback về `source_dataset` trong JSON), thực thi truy vấn SQL lấy toàn bộ dữ liệu, sau đó gọi `_convert_data_types` (của lớp cha) để làm phẳng JSON và chuẩn hóa kiểu dữ liệu.
    *   **`process_and_split_data(df)`**: Tiến hành loại bỏ tự động các cột rác (được định nghĩa trong `columns_to_drop` của YAML). Sau đó thực hiện phân chia dữ liệu thành 3 tập: Train, Validation và Test dựa theo tỷ lệ cấu hình an toàn (random_state=42).
    *   **`export_to_feature_store(...)`**: Xuất các DataFrames (Train/Val/Test/Current) ra định dạng `.csv` lưu vào thư mục Feature Store. Đặc biệt, nếu cấu hình `dvc_tracked` được kích hoạt, phương thức sẽ tự động gọi `self.dvc.add_data(file_path)` để phiên bản hóa dữ liệu ngay lúc lưu.
    *   **`run()`**: Hàm entry point thực thi toàn bộ quy trình. Khởi động một phiên theo dõi trên MLflow bằng `self.tracker.start_run(run_name="data_ingestion")` để quản lý luồng dữ liệu.
    *   *Giám sát hiệu năng:* Tất cả các phương thức cốt lõi (extract, process, export) đều được bọc bởi decorator `@monitor_performance()` nhằm ghi nhận thời gian chạy vào log hệ thống tự động.

*File:* [data_val.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/pipelines/ml_pipelines/calories_pred/data_val.py)
*   **`load_class_from_registry(api_path: str)`**: Hàm tiện ích giúp nạp động (dynamic import) các lớp kiểm định (schema classes) dựa trên đường dẫn cấu hình trong file YAML. Cơ chế này giúp loại bỏ việc hard-code các import trực tiếp, tăng tính linh hoạt khi thay đổi rule kiểm định.
*   **`run_validation_pipeline(...)`**: Hàm lõi thực thi quy trình kiểm định MLOps sử dụng Pandera (với cấu hình `lazy=True` để quét toàn bộ lỗi thay vì dừng ở lỗi đầu tiên).
    *   **Cơ chế Quarantine & Rescue (Cách ly và Cứu vãn)**: Tự động tách các bản ghi vi phạm logic (failure_cases) lưu vào thư mục `quarantine_dir` cùng file báo cáo chi tiết. Phần dữ liệu hợp lệ (valid_df) vẫn được "cứu" và lưu trữ dưới dạng `.parquet` cho bước Transformation tiếp theo.
    *   **Alerting & Thresholds**: Tính toán tỷ lệ lỗi (`error_rate`). Nếu tỷ lệ này vượt qua ngưỡng cho phép (`error_threshold`), hệ thống lập tức ghi log Error, phát cảnh báo và ngắt Pipeline bằng Exception.
    *   **Tích hợp MLflow**: Tự động theo dõi các metrics liên quan đến chất lượng dữ liệu như `val_{mode}_error_rate` và `val_{mode}_invalid_count` (với mode là train, val, hoặc test) vào MLOpsTracker.
*   **`CaloriesDataValidationPipeline`**: Lớp điều phối (Orchestrator) tổng thể. Chịu trách nhiệm nạp cấu hình từ `data_val.yaml`, khởi tạo các thư mục (input, output, quarantine), khởi chạy `MLOpsTracker` và duyệt qua toàn bộ các file `.parquet` thô để tiến hành kiểm định hàng loạt.

*File:* [eval_drift.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/pipelines/ml_pipelines/calories_pred/eval_drift.py)
*   **`load_yaml_config(path: str)`**: Hàm trợ giúp đọc file cấu hình YAML, tích hợp bắt lỗi và ghi log khi file không tồn tại.
*   **`CaloriesDriftEvaluator`**: Lớp cốt lõi đánh giá độ trôi lệch dữ liệu (Data Drift), thiết kế đọc tham số động từ các file cấu hình `evidently_profile.yaml` và `global.yaml`.
    *   **Khởi tạo (`__init__`)**: Tự động trích xuất các đường dẫn I/O (dữ liệu tham chiếu, thư mục xuất report), các ngưỡng cảnh báo MLOps (`psi_warning`, `psi_critical`, `p_value_alpha`) và ánh xạ cột (`column_mapping`).
    *   **`evaluate_drift(current_df, reference_df)`**: Phương thức đo lường chính, được giám sát hiệu năng với decorator `@monitor_performance(time_limit=10.0)`. Sử dụng `EvidentlyDriftEngine` để xuất báo cáo HTML và JSON. Trích xuất tỷ lệ trôi lệch (`drift_share`) và đối chiếu với các Thresholds để phát ra cảnh báo:
        *   **CRITICAL ALERT**: Vượt ngưỡng `psi_critical` (yêu cầu Retrain Model).
        *   **WARNING**: Vượt ngưỡng `psi_warning` (cần theo dõi).
        *   **INFO**: Dữ liệu ổn định.
*   **`run_drift_detection(...)`**: Hàm Wrapper (đóng gói) tiện ích. Cho phép Data Validation Pipeline hoặc Orchestrator gọi thẳng vào quy trình đánh giá Drift một cách gọn gàng, ẩn đi các chi tiết khởi tạo class và cấu hình bên dưới.

## 6. Data Validation (Hệ sinh thái Pandera & evidently AI)

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

## 7. Database & CRUD ( src/nutrition_core/database )
Phân hệ quản lý kết nối cơ sở dữ liệu, khai báo mô hình (ORM) và các thao tác truy vấn (CRUD).

*File:* [`src/nutrition_core/database/session.py`]
- Cung cấp `Base` (declarative_base) và quản lý session của SQLAlchemy.
- Thiết lập connection pool để tương tác với cơ sở dữ liệu PostgreSQL.

*File:* [`src/nutrition_core/database/models.py`]
- `build_model_from_yaml(yaml_filename)`: Đọc tệp cấu hình YAML (từ `config/datasets/`) và tự động đăng ký Model động vào `Base.metadata`. Hàm này hỗ trợ ánh xạ tự động các kiểu dữ liệu cơ bản (integer, string, float, date...) từ cấu hình YAML sang SQLAlchemy Types.

*Thư mục:* [`src/nutrition_core/database/crud/`]

- Chứa các module xử lý thao tác Create, Read, Update, Delete.
- Gồm các tệp chuyên biệt như `user_crud.py` (quản lý người dùng) và `vector_crud.py` (quản lý cơ sở dữ liệu vector/RAG).

## 8. Database, EDA & Automation Scripts ( src/scripts )

Phân hệ này chứa các đoạn mã chạy độc lập (standalone scripts) phục vụ cho việc tự động hóa cơ sở dữ liệu PostgreSQL, trích xuất báo cáo phân tích và sinh tự động các file cấu hình schema.

*File:* [run_db_bulk_load.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/scripts/run_db_bulk_load.py)
- **Chức năng**: Quản lý tiến trình nạp dữ liệu hàng loạt (bulk load) vào cơ sở dữ liệu PostgreSQL.
- **Điểm nhấn kiến trúc**: Loại bỏ hoàn toàn việc mã hóa cứng (hardcode) đường dẫn. Được thiết kế để đọc siêu dữ liệu (metadata) và định nghĩa bảng trực tiếp từ cấu hình động `schema.yaml` và `docs/architecture/tables_created_report.yaml`.

*File:* [create_tables.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/scripts/create_tables.py)
- **Chức năng**: Khởi tạo tự động các bảng trong PostgreSQL thông qua SQLAlchemy dựa trên định nghĩa từ file YAML.
- **Điểm nhấn kiến trúc**: Hàm `get_pg_data_type()` tự động ánh xạ các kiểu dữ liệu từ YAML (như `string`, `int`, `boolean`) sang kiểu chuẩn của PostgreSQL (như `VARCHAR(255)`, `BIGINT`, `BOOLEAN`).

*File:* [create_schema_pipelines.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/scripts/create_schema_pipelines.py)
- **Chức năng**: Đọc cấu hình từ `tables_created_report.yaml` để phân luồng và tạo tự động các cấu hình schema pipeline tương ứng cho từng bộ dữ liệu đặc thù.

*File:* [eda_report.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/scripts/eda_report.py)
- **Chức năng**: Trích xuất báo cáo phân tích dữ liệu khám phá (EDA) kết hợp sử dụng `ydata_profiling`.
- **Điểm nhấn kiến trúc**: Tích hợp hàm `calculate_custom_metrics()` để đo lường chính xác các chỉ số từ table-level, numerical, categorical đến relationships. Luồng xử lý phân tích tuân thủ nghiêm ngặt nguyên tắc **không sử dụng IQR để loại bỏ outlier** nhằm bảo toàn tính toàn vẹn của các điểm bất thường (anomalies) có ý nghĩa thực tế trong dữ liệu sức khỏe và dinh dưỡng.

*File:* [generate_data_schema.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/scripts/generate_data_schema.py)
- **Chức năng**: Tự động sinh ra các bộ khung (skeleton) schema cho hệ sinh thái Pandera dựa trên file JSON Metrics được xuất ra từ quá trình chạy `eda_report.py`. Phục vụ trực tiếp cho lớp Data Validation.

## 10. API Framework ( src/api )

