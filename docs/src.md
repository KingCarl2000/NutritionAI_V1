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
*Mã nguồn:* [src/postgres/monitoring/query_analyzer.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/postgres/monitoring/query_analyzer.py)

*   **`analyze_query_performance(query: str)`**: Công cụ phân tích và đo lường hiệu năng của các câu lệnh SQL truy vấn vào Feature Store.
    *   **Cơ chế hoạt động**: Tự động chèn tiền tố `EXPLAIN (ANALYZE, FORMAT JSON)` vào truy vấn nguyên bản để phân tích cấu trúc thực thi.
    *   **Thực thi an toàn**: Quản lý truy vấn an toàn thông qua việc ép kiểu với `psycopg.sql.SQL` và `typing.LiteralString`, sử dụng luồng kết nối trực tiếp từ `get_connection()`.
    *   **Giám sát hiệu suất**: Trích xuất kết quả phân tích (execution plan) dưới dạng JSON để lấy thông tin chi tiết về thời gian thực thi thực tế, đồng thời hỗ trợ phát hiện các tình huống rủi ro suy giảm hiệu suất của Database như quét toàn bảng (Sequential Scan / Seq Scan).
    *   **Cảnh báo tự động**: Tích hợp custom logger để xuất thông báo (mức `warning`) nếu không thể trích xuất dữ liệu execution plan hợp lệ từ PostgreSQL.

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

### A. API

*Mã nguồn:* [src/nutrition_core/pandera/api/base.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/pandera/api/base.py)

*   **`BaseSchema(pa.DataFrameModel)`**: Lớp Schema nền tảng áp dụng các quy tắc kiểm định khung. Tất cả các schema nghiệp vụ phía sau (như `CaloriesSchema`, `UserSchema`) đều phải kế thừa từ lớp này để đồng bộ hóa quy tắc xử lý.
    *   **Cấu hình hệ thống (Class Config)**: Định nghĩa các quy tắc cốt lõi:
        *   `coerce = True`: Tự động ép kiểu dữ liệu của DataFrame về đúng định dạng (dtype) đã khai báo.
        *   `strict = False` & `ordered = False`: Thiết kế linh hoạt, cho phép DataFrame chứa các cột phụ (không định nghĩa trong schema) và không ép buộc thứ tự các cột.
        *   `unique_column_names = True`: Đảm bảo tính toàn vẹn, ngăn chặn các cột bị trùng lặp tên.
        *   `add_missing_columns = True`: Tự động bổ sung các cột bị thiếu và điền giá trị mặc định (`default`) nếu cột đó được cấu hình là không bắt buộc.
    *   **`validate_df(cls, df, lazy, drop_invalid_rows)`**: Giao diện (Interface) chính để thực thi kiểm định.
        *   Sử dụng trình quản lý tùy chỉnh **`NutritionPandasBackend`** làm Orchestrator nhằm bóc tách logic duyệt và lọc dữ liệu lỗi khỏi Pandera gốc.
        *   Tích hợp decorator `@validation_error_handler()` để tự động chặn các ngoại lệ `SchemaError`/`SchemaErrors` và ghi trực tiếp vào `error_logs`.
        *   Tự động gắn kết với hệ thống theo dõi `MLOpsTracker` thông qua experiment `Data_Validation_Pipeline`, giúp dễ dàng truy vết trạng thái kiểm định trên bảng điều khiển MLflow.

*Mã nguồn:* [src/nutrition_core/pandera/api/check.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/pandera/api/check.py)

Module này đóng vai trò là một thư viện tiện ích (Utilities) độc lập với dataset, cung cấp các quy tắc kiểm định (Checks) nâng cao để các lớp Schema trong hệ thống có thể dễ dàng gọi và tái sử dụng.

*   **1. Extensions API (Phương thức Vectorized mở rộng)**:
    *   Sử dụng decorator `@pa_ext.register_check_method` để nhúng trực tiếp các hàm kiểm tra hiệu năng cao (vectorized) vào thư viện Pandera.
    *   **`is_in_numeric_range`**: Đảm bảo các giá trị trong một cột số nằm đúng trong giới hạn `[min_val, max_val]`.
    *   **`is_valid_string_length`**: Đảm bảo độ dài của chuỗi ký tự nằm trong giới hạn cho phép `[min_len, max_len]`.
*   **2. `GenericCheckFactory` (Lớp khởi tạo quy tắc linh hoạt)**:
    *   Tập hợp các phương thức tĩnh (`@staticmethod`) tạo quy tắc kiểm định đa cột hoặc theo nhóm được tham số hóa:
    *   **`check_sum_le_total`** *(Wide Check)*: Ràng buộc logic đảm bảo tổng giá trị của các cột thành phần (`component_cols`) nhỏ hơn hoặc bằng cột tổng (`total_col`).
    *   **`check_range_validity`** *(Wide Check)*: Đảm bảo tính hợp lệ của chuỗi dữ liệu theo thứ tự nghiêm ngặt: `min_col <= target_col <= max_col`.
    *   **`check_group_mean_comparison`** *(Groupby Check)*: Gom nhóm dữ liệu (`groupby_col`) và so sánh giá trị trung bình, đảm bảo nhóm thấp (`lower_group`) có giá trị trung bình nhỏ hơn nhóm cao (`higher_group`).
    *   **`check_regex_pattern`** *(Element-wise Check)*: Đối chiếu từng giá trị đơn lẻ trong cột xem có khớp với biểu thức Regex truyền vào hay không.
    *   **`create_outlier_warning`** *(Soft Check)*: Cơ chế kiểm định "mềm". Cấu hình `raise_warning=True` để phát ra tín hiệu `SchemaWarning` ghi log khi phát hiện giá trị lớn hơn `upper_threshold` (Outlier) thay vì văng Exception làm ngắt ngang toàn bộ Pipeline luân chuyển dữ liệu.

### B. Pandas Execution Backend (Thực thi kiểm định hệ thống)
*Mã nguồn:* [src/nutrition_core/pandera/backends/pandas_executor.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/pandera/backends/pandas_executor.py)

Module này chứa Engine thực thi lõi, đảm nhiệm việc bóc tách logic kiểm định khỏi Pandera gốc, giúp hệ thống kiểm soát chặt chẽ quá trình tiền xử lý, bẫy lỗi và cách ly dữ liệu hỏng đối với cấu trúc dữ liệu Pandas DataFrame.

*   **`NutritionPandasBackend`**: Lớp điều phối trung gian (Facade) nhận cấu hình từ các Schema (như `lazy`, `drop_invalid_rows`) và quản lý vòng đời kiểm định:
    *   **`_preprocess_and_coerce(df)`**: Bước tiền xử lý chạy trước Pandera. Khớp tên cột (hỗ trợ cả Regex), kiểm tra thiếu sót các cột bắt buộc, và chủ động ép kiểu dữ liệu (coerce). Các lỗi ép kiểu sẽ được gom lại thay vì làm sập chương trình.
    *   **`_execute_vectorized_checks(df)`**: Thực thi bộ quy tắc kiểm định cốt lõi. Nếu có vi phạm, phương thức bẫy ngoại lệ `SchemaErrors`, gọi `PanderaErrorHandler` (từ `val_exception.py`) để trích xuất lỗi thành danh sách dictionary, đồng thời tạo ra một `valid_mask` (Boolean Series) để đánh dấu chính xác các dòng lỗi.
    *   **`run_pipeline(df)`**: Luồng thực thi tổng quát (Orchestrator). Trình tự: Xác thực Backend -> Tiền xử lý -> Chạy Checks -> Tự động loại bỏ dòng vi phạm (nếu `drop_invalid_rows=True`) -> Trả về DataFrame "sạch" (`valid_df`) cùng danh sách các lỗi (`failure_cases`). Trong trường hợp `lazy=False`, tự động ném ra `DataValidationError` đã được chuẩn hóa hệ thống.
*   **`validate_with_pandas_backend(schema, df, **kwargs)`**: Hàm Wrapper API

### D. Engines

*Mã nguồn:* [src/nutrition_core/pandera/decorators.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/pandera/decorators.py)

Module chứa các Decorators chuyên dụng dùng để bọc (wrap) các decorator gốc của thư viện Pandera. Mục tiêu của phân hệ này là tự động hóa việc chuẩn hóa ngoại lệ và gắn kết luồng kiểm định với hệ thống theo dõi MLOps mà không làm thay đổi logic gốc của các hàm xử lý dữ liệu.

*   **Tích hợp MLOps (MLflow Tracker)**: Module tự động khởi tạo một phiên bản `MLOpsTracker` với experiment name là `Data_Validation_Pipeline` để quản lý tập trung toàn bộ tiến trình.
*   **`_log_validation_failure(func_name, errors)`**: Hàm nội bộ (helper) làm nhiệm vụ gọi hook `log_validation_failure` của tracker để đẩy trực tiếp danh sách chi tiết các lỗi kiểm định lên bảng điều khiển MLflow ngay khi pipeline gặp sự cố.
*   **`@nutrition_check_types(*args, **kwargs)`**:
    *   Mở rộng từ `pandera.check_types`. Thực hiện kiểm định dữ liệu tự động dựa trên Type Annotations (Type Hint) của hàm (ví dụ: tham số truyền vào được đánh dấu là `DataFrame[SchemaModel]`).
    *   **Cơ chế bắt lỗi an toàn**: Bẫy các ngoại lệ `SchemaError` hoặc `SchemaErrors` nguyên thủy từ Pandera. Sau đó, ghi log (Error) chi tiết tên hàm bị sụp, gọi hàm `_log_validation_failure` để gửi báo cáo lỗi, và cuối cùng ném ra ngoại lệ chuẩn của hệ thống là `DataValidationError` để ngắt pipeline an toàn.
*   **`@nutrition_check_io(head=5, tail=5, **kwargs)`**:
    *   Mở rộng từ `pandera.check_io`. Dùng để kiểm định tường minh dữ liệu đầu vào (Input) và đầu ra (Output) của một hàm thông qua `DataFrameSchema` (ví dụ: `df=schema_in`, `out=schema_out`).
    *   Kế thừa toàn bộ cơ chế bắt lỗi, trích xuất lỗi qua `PanderaErrorHandler`, ghi log hệ thống và tracking MLOps tương tự như `nutrition_check_types`.

*Mã nguồn:* [src/nutrition_core/pandera/config.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/pandera/config.py)

Module quản lý cấu hình toàn cục cho quá trình kiểm định dữ liệu của Pandera trong hệ thống, giúp tinh chỉnh linh hoạt giữa các môi trường triển khai (Dev, Staging, Production).

*   **`setup_pandera_validation(mlflow_tracker=None)`**: Hàm khởi tạo cấu hình trung tâm, được thiết kế để gọi một lần tại điểm khởi chạy (entry point) của Data Validation Pipeline.
    *   **Quản lý Validation Depth (Độ sâu kiểm định)**: Tự động nạp cấu hình từ biến môi trường `PANDERA_VALIDATION_DEPTH`. Nó ánh xạ các giá trị chuỗi sang `ValidationDepth` Enum của Pandera: `SCHEMA_ONLY`, `DATA_ONLY`, hoặc `SCHEMA_AND_DATA` (mặc định). Tính năng này cho phép bạn dễ dàng tắt bớt việc quét Data (chỉ giữ lại quét Schema) để tiết kiệm thời gian xử lý khi chạy trên Production với lượng dữ liệu khổng lồ.
    *   **Tích hợp Tracking MLOps**: Nếu hàm được cấp một đối tượng `MLOpsTracker`, nó sẽ thực hiện gọi `log_params_and_metrics` để ghi nhận trực tiếp thông số `pandera_validation_depth` lên bảng điều khiển MLflow, giúp quản lý lịch sử cấu hình của từng phiên chạy.
    *   **Giám sát hiệu năng khởi tạo**: Sử dụng decorator `@monitor_performance(time_limit=2.0)` để đảm bảo rằng tác vụ đọc biến môi trường và thiết lập ngữ cảnh không vượt quá 2 giây, tránh hiện tượng thắt nút cổ chai (bottleneck) ngay từ đầu luồng.
    *   **Thiết lập Ngữ cảnh**: Hàm trả về đối tượng `config_context(validation_depth=...)`, giúp áp dụng giới hạn kiểm định này vào phạm vi hoạt động của toàn bộ luồng xử lý phía sau một cách đồng nhất.

*Mã nguồn:* [`src/nutrition_core/pandera/evidently_engine.py`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/pandera/evidently_engine.py)

Lớp điều phối trung gian (`EvidentlyDriftEngine`) tích hợp đồng thời Pandera Schema Validation, Evidently AI Data Drift Engine, DVC Artifact Management và hệ thống MLflow Logging cho hệ thống [NutritionAI_V1](https://github.com/KingCarl2000/NutritionAI_V1).

*   **Khởi tạo (`__init__`)**: Nhận các tham số cốt lõi gồm `artifact_dir`, `project_name`, `evidently_workspace_url` và `mlops_tracker`. Tự động khởi tạo thư mục lưu trữ báo cáo (ví dụ: `Artifacts/reports/ml_calories/evidently`) nếu chưa tồn tại.
*   **Các giai đoạn thực thi chính**:
    *   **Giai đoạn 1 - `validate_schema(df, schema_model)`**: Thực thi kiểm tra cấu trúc lược đồ dữ liệu động sử dụng mô hình Pandera Schema trước khi đưa vào phân tích trôi lệch. Trả về trạng thái `True/False` kèm theo DataFrame đã được xác thực.
    *   **Giai đoạn 2 - `run_drift_detection(current_df, reference_df, column_mapping_cfg, drift_options_cfg)`**: 
        *   Trích xuất linh hoạt các đặc trưng số (`numerical_columns`), phân loại (`categorical_columns`) và văn bản (`text_columns`) từ file cấu hình YAML để xây dựng `DataDefinition` và `Dataset` của Evidently.
        *   Thiết lập cấu hình kiểm định trôi lệch (`DataDriftPreset`) và tóm tắt dữ liệu (`DataSummaryPreset`) dựa trên các ngưỡng cấu hình (`dataset_drift_share`, `p_value_alpha`, `wasserstein_distance`, v.v.).
        *   Xuất và lưu trữ báo cáo dưới dạng tệp `drift_report.html` và `drift_metrics.json` vào thư mục Artifacts.
        *   Đồng bộ snapshot trực tiếp lên Evidently Remote Workspace UI (nếu có cấu hình `evidently_workspace_url`).
    *   **Giai đoạn 3 - `log_metrics_to_mlflow(validation_json)`**: 
        *   Trích xuất các chỉ số quan trọng từ báo cáo JSON như `drift_share`, `number_of_drifted_columns` và cờ cảnh báo `dataset_drift_detected`.
        *   Sử dụng đối tượng `MLOpsTracker` để đẩy metrics lên MLflow và tự động tải các tệp báo cáo HTML/JSON làm hệ thống artifact logging để dễ dàng truy vết lịch sử vận hành mô hình.

## 7. Database & CRUD (`src/nutrition_core/database`)

Phân hệ này quản lý việc định nghĩa các cấu trúc bảng (ORM Models) và phiên giao dịch (Sessions) để tương tác với cơ sở dữ liệu. Điểm đặc biệt của kiến trúc này là khả năng sinh Model động từ file cấu hình thay vì hard-code.

### A. Session Management (Quản lý phiên kết nối)
*Mã nguồn:* [src/nutrition_core/database/session.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/database/session.py)

*   **Khởi tạo `SessionLocal`**: Khởi tạo nhà máy tạo Session (`sessionmaker`) của SQLAlchemy để cấp phát các phiên làm việc với cơ sở dữ liệu.
*   **Đảm bảo tính toàn vẹn (ACID)**: Cấu hình `autocommit=False` và `autoflush=False` giúp hệ thống kiểm soát chặt chẽ thời điểm xác nhận (commit) hoặc hoàn tác (rollback) dữ liệu, tránh tình trạng ghi dữ liệu rác khi các pipeline xảy ra lỗi đột ngột.
*   **Tích hợp Engine & Logging**: Tự động kết nối với hàm `get_engine()` (từ `src.postgres.core.connection`) và tích hợp custom logger để theo dõi toàn bộ trạng thái khởi tạo.

### B. Dynamic Modeling (Định nghĩa Model động)
*Mã nguồn:* [src/nutrition_core/database/models.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/nutrition_core/database/models.py)

Phân hệ này được thiết kế để tự động hóa hoàn toàn việc tạo cấu trúc bảng ORM dựa trên các định nghĩa khai báo trong file cấu hình YAML.

*   **`TYPE_MAPPING`**: Từ điển ánh xạ các chuỗi định nghĩa kiểu dữ liệu trong YAML (như `"integer"`, `"string"`, `"boolean"`, `"date"`, `"datetime"`) sang các lớp kiểu dữ liệu chuẩn tương ứng của SQLAlchemy (`Integer`, `String`, `Boolean`, `Date`, `DateTime`).
*   **`build_model_from_yaml(yaml_filename: str)`**: Hàm cốt lõi chịu trách nhiệm sinh Model động.
    *   **Cơ chế hoạt động**: Nhận tên file YAML (nằm trong thư mục `config/datasets/`), tự động đọc và phân tích cấu trúc, bao gồm cả việc xử lý các dictionary lồng nhau (nested dictionary) và các thuộc tính cột.
    *   **Đăng ký tự động**: Các Model (Lớp) sinh ra sẽ tự động được trả về và đăng ký vào `Base.metadata`, giúp hệ thống ORM (như Alembic) có thể nhận diện ngay lập tức để thực hiện migration.
    *   **Bắt lỗi cấu hình**: Nếu không tìm thấy file cấu hình được chỉ định, hàm sẽ ghi log lỗi ở mức ERROR và ném ra ngoại lệ `FileNotFoundError`, ngăn chặn hệ thống chạy với một database rỗng.

## 8. Automation Scripts (`src/scripts`)

Phân hệ này chứa các đoạn mã tự động hóa quy trình vận hành hệ thống, giúp kỹ sư chạy các tác vụ Data Ingestion, khởi tạo Database, hoặc luồng MLOps một cách độc lập và hàng loạt từ Command Line.

### A. Tự động hóa Bulk Load (Nạp dữ liệu lớn)
*Mã nguồn:* [`src/scripts/run_db_bulk_load.py`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/scripts/run_db_bulk_load.py)

*   **Định tuyến động (`PROJECT_ROOT`)**: Tự động xác định đường dẫn gốc của dự án bằng `pathlib.Path` và đưa vào `sys.path`, cho phép thực thi script từ bất kỳ thư mục nào mà không bị lỗi module import.
*   **YAML-Driven Ingestion (Nạp qua cấu hình)**: 
    *   Hàm `main(config_rel_path)` đọc trực tiếp các thông số nạp dữ liệu từ file YAML (ví dụ: `config/datasets/calories.yaml`).
    *   Hỗ trợ duyệt qua nhiều bảng (`dataset_key`, `dataset_info`) được định nghĩa trong cùng một file cấu hình để thực thi nạp hàng loạt.
*   **Tích hợp `PostgresBulkLoader`**: Khởi tạo bộ nạp dữ liệu và gọi hàm `execute_bulk_load` cho từng file `.csv` thô, tận dụng cơ chế `COPY TO STDIN` của PostgreSQL để tối ưu tốc độ nạp thay vì dùng lệnh INSERT.
*   **Thiết kế dễ mở rộng**: Khối `if __name__ == "__main__":` duy trì một danh sách `config_files`. Kỹ sư chỉ cần mở comment (ví dụ: `"config/datasets/recipes.yaml"`) để luồng script tự động quét và nạp thêm các Domain dữ liệu mới vào DB.

### B. Tự động hóa Khởi tạo Bảng (Database Schema Creation)
*Mã nguồn:* [src/scripts/create_tables.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/scripts/create_tables.py)

Script này chịu trách nhiệm đọc cấu hình lược đồ dữ liệu và tự động sinh ra các bảng vật lý trong cơ sở dữ liệu PostgreSQL, loại bỏ hoàn toàn việc phải viết các câu lệnh SQL DDL thủ công.

*   **Ánh xạ kiểu dữ liệu (`get_pg_data_type`)**: Tự động chuyển đổi các kiểu dữ liệu chung được khai báo trong file YAML (như `string`, `int`, `boolean`, `datetime`) sang các kiểu dữ liệu chuẩn tối ưu của PostgreSQL (như `VARCHAR(255)`, `BIGINT`, `BOOLEAN`, `TIMESTAMP`).
*   **Chuẩn hóa định danh (`normalize_identifier`)**: Xử lý và làm sạch tên các tập dữ liệu (dataset) để tạo ra các tiền tố (prefix) bảng an toàn cho PostgreSQL, ngăn chặn các lỗi cú pháp do ký tự đặc biệt hoặc khoảng trắng gây ra.
*   **Luồng thực thi lõi (`create_tables_from_yaml`)**:
    *   Kết nối an toàn với Database thông qua `get_connection()`.
    *   Tự động sinh và thực thi chuỗi lệnh DDL: Đảm bảo Schema tồn tại (`CREATE SCHEMA IF NOT EXISTS`), dọn dẹp bảng cũ (`DROP TABLE IF EXISTS CASCADE`), và khởi tạo bảng mới (`CREATE TABLE`).
    *   **Cơ chế Báo cáo (Report Generation)**: Sau khi tạo bảng thành công, script sẽ tự động tổng hợp và xuất ra file báo cáo `tables_created_report.yaml` (lưu tại `docs/architecture/`). File báo cáo này đóng vai trò cực kỳ quan trọng làm "nguồn sự thật" (Source of Truth) đầu vào cho các luồng Ingestion hoặc Automation Scripts khác (như `run_db_bulk_load.py`) biết chính xác bảng nào đã sẵn sàng để nạp dữ liệu.

### C. Tự động hóa Tạo Schema Pipeline (Pipeline Schema Generation)
*Mã nguồn:* [src/scripts/create_schema_pipelines.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/scripts/create_schema_pipelines.py)

Script này đóng vai trò cầu nối dữ liệu (Data Bridge) giữa hệ thống Cơ sở dữ liệu và các luồng Machine Learning (ML Pipelines). Nó đọc báo cáo cấu trúc bảng đã khởi tạo và tự động sinh ra các file cấu hình JSON chuyên biệt (ví dụ: `calories_data_schema.json`) phục vụ cho khâu trích xuất và biến đổi dữ liệu.

*   **Kế thừa Nguồn sự thật (Single Source of Truth)**: Tự động đọc file `docs/architecture/tables_created_report.yaml` (được sinh ra từ bước tạo bảng). Việc này đảm bảo luồng ML luôn đồng bộ tuyệt đối với cấu trúc vật lý của PostgreSQL mà không cần kỹ sư phải khai báo lại cấu trúc bằng tay.
*   **Phân loại Cột Tự động (Smart Column Categorization)**: Dựa vào thuộc tính `yaml_type`, script tự động phân tách và gom nhóm các đặc trưng thành:
    *   `numerical_cols`: Các cột có kiểu `integer` hoặc `float` (dùng cho Standard Scaling, tính toán độ lệch).
    *   `categorical_cols`: Các cột có kiểu `categorical` (dùng cho One-Hot Encoding hoặc Entity Embedding).
*   **Kiến trúc Đăng ký Mở rộng (`PIPELINES_REGISTRY`)**: Quản lý tập trung các cấu hình pipeline thông qua một danh sách dictionary. Để tích hợp một domain dữ liệu mới (ví dụ: `recipes`), kỹ sư chỉ cần khai báo một block mới trong biến `PIPELINES_REGISTRY` (gồm `source_dataset`, `dir_path`, `index`, `drop_cols`). Script sẽ tự động quét và render file JSON cấu hình tương ứng.
*   **Tích hợp đầu ra JSON**: Gói toàn bộ metadata (cột index, các cột cần loại bỏ, phân loại numerical/categorical) thành một file `.json` hoàn chỉnh. File này sau đó sẽ được `BasePostgresExtractor` và các module Data Ingestion nạp vào để chuẩn hóa DataFrame tự động.

### D. Tự động hóa Báo cáo EDA (EDA Report Generation)
*Mã nguồn:* [src/scripts/eda_report.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/scripts/eda_report.py)

Script này tự động hóa quá trình phân tích khám phá dữ liệu (Exploratory Data Analysis - EDA), xuất ra các báo cáo trực quan và số liệu thống kê chi tiết cho từng bảng dữ liệu trong PostgreSQL.

*   **Quản lý cấu hình động**: Tự động đọc định nghĩa các bảng cần phân tích từ file cấu hình `schema.yaml` (nằm trong `src/api/data_schema/`).
*   **Trích xuất mẫu dữ liệu an toàn**: Tận dụng `get_connection()` từ Connection Pool để truy vấn dữ liệu từ PostgreSQL. Sử dụng giới hạn (`LIMIT 10000`) thông qua SQL để tránh tình trạng quá tải bộ nhớ (OOM) khi phân tích các bảng dữ liệu khổng lồ.
*   **Tính toán Metric tùy chỉnh (`calculate_custom_metrics`)**: 
    *   **Table-level**: Thống kê số dòng, số cột, tỷ lệ dữ liệu thiếu (missing) và dữ liệu trùng lặp (duplicate).
    *   **Numerical metrics**: Tính toán chuyên sâu cho dữ liệu số (min, max, mean, median, IQR, std_dev, skewness, kurtosis, zero ratio).
    *   **Categorical/Datetime metrics**: Đo lường tần suất (mode, frequency), số lượng giá trị duy nhất (distinct count) và tự động nhận diện ép kiểu thời gian.
    *   **Relationships**: Tự động tính toán ma trận tương quan theo phương pháp Pearson và Spearman.
*   **Kết xuất báo cáo kép (Dual Output)**: Kết quả phân tích được lưu tự động vào thư mục lưu trữ tĩnh `Artifacts/EDA_reports/` dưới 2 định dạng riêng biệt:
    1.  **Cấu trúc JSON (`{table_name}_metrics.json`)**: Lưu trữ các custom metrics dưới dạng JSON để tiện cho việc tái sử dụng bằng máy trong các luồng MLOps tiếp theo.
    2.  **Trực quan HTML (`{table_name}_dataset_schema.html`)**: Báo cáo trực quan sinh động, chứa đồ thị tương tác được tạo tự động bởi thư viện `ydata-profiling` (`ProfileReport`).

### E. Tự động hóa Sinh khung kiểm định (Schema Skeleton Generation)
*Mã nguồn:* [src/scripts/generate_data_schema.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/scripts/generate_data_schema.py)

Script này đóng vai trò tự động hóa việc tạo bộ khung (skeleton) cho các file cấu hình Pandera Schema, giúp giảm thiểu tối đa thao tác viết mã thủ công lặp đi lặp lại (boilerplate code) cho Data Engineer.

*   **Kế thừa Metrics tĩnh**: Tự động đọc và phân tích cấu trúc file `{table_name}_metrics.json` (được sinh ra từ bước chạy `eda_report.py`) để trích xuất chính xác danh sách các cột dạng số (numerical) và phân loại/thời gian (categorical/datetime).
*   **Tự động sinh mã nguồn (Code Generation)**:
    *   Dựa vào tên pipeline (`pipeline_name`), script tự động chuẩn hóa tên Class (ví dụ: `ml_calories` thành `MlCaloriesInputSchema`).
    *   Render ra một file mã nguồn Python (`.py`) hoàn chỉnh định nghĩa các lớp Schema kế thừa từ `pa.DataFrameModel`.
    *   Thiết lập sẵn các cấu hình bảo vệ cốt lõi: `coerce = True`, `strict = True` và đặt mặc định các trường là bắt buộc (`nullable=False`).
*   **Phân luồng Target Column**: Tự động nhận diện cột mục tiêu dự đoán (`target_col`) để tách biệt kiến trúc. Cột mục tiêu sẽ chỉ được khai báo trong `TrainingSchema` (kế thừa từ `InputSchema`), giữ cho `InputSchema` thuần túy chỉ chứa các đặc trưng đầu vào (features).
*   **Cơ chế Human-in-the-loop (Chờ chuyên gia tinh chỉnh)**: 
    *   Script chỉ đóng vai trò tạo khung định kiểu cơ bản (`float` cho số, `str` cho chuỗi).
    *   Nó tự động chèn các ghi chú `TODO` bên cạnh mỗi thuộc tính (Ví dụ: `TODO: Xác nhận lại kiểu (int/float) và cập nhật ngưỡng` hoặc `TODO: Cập nhật danh sách isin=[...]`). Điều này định hướng trực tiếp cho Data Engineer/Domain Expert vị trí cần can thiệp thủ công để điền các quy tắc logic nghiệp vụ và ngưỡng sinh học chính xác.
*   **Thực thi hàng loạt (Batch Execution)**: Khối cấu hình trung tâm `ML_PIPELINES_CONFIG` cho phép ánh xạ 1-1 giữa nhiều file JSON metrics và đường dẫn xuất schema đầu ra, dễ dàng sinh khung cho hàng loạt Domain cùng lúc.

### F. Tự động hóa cấu hình DVC Pipelines (DVC Stages Setup)
*Mã nguồn:* [src/scripts/setup_dvc_stages.sh](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/src/scripts/setup_dvc_stages.sh)

Script Bash này tự động hóa việc khởi tạo và liên kết các bước (stages) trong luồng máy học thành một pipeline hoàn chỉnh thông qua Data Version Control (DVC). Việc chạy script này sẽ tạo/cập nhật file `dvc.yaml`, đảm bảo tính tái lập (reproducibility) cho toàn bộ quy trình MLOps của phân hệ Calories Prediction.

*   **Tự động định vị thư mục (Auto-Navigation)**: Sử dụng lệnh `cd "$(dirname "$0")/../.."` để luôn di chuyển luồng thực thi về thư mục gốc của dự án `NutritionAI_V1`. Điều này cực kỳ quan trọng giúp DVC nhận diện chính xác các đường dẫn tương đối (relative paths) của dữ liệu và mã nguồn, bất kể bạn gọi script từ thư mục nào.
*   **Định nghĩa 5 giai đoạn cốt lõi (ML Pipeline Stages)**: Script thiết lập 5 stage nối tiếp nhau, đầu ra (`-o`) của bước trước làm đầu vào (`-d`) cho bước sau:
    1.  **`calories_ingest` (Trích xuất dữ liệu)**: Theo dõi mã nguồn `data_ingest.py` và cấu hình YAML. Xuất ra file khung `calories_data_schema.json` và dữ liệu thô tại `Artifacts/data/ml_calories/raw`.
    2.  **`calories_validate` (Kiểm định chất lượng)**: Nhận dữ liệu thô từ bước 1. Theo dõi `data_val.py` và `calories_schema.py`. Phân tách đầu ra thành dữ liệu sạch (`validated`) và dữ liệu lỗi (`quarantine`).
    3.  **`calories_transform` (Biến đổi dữ liệu)**: Dựa trên báo cáo kiểm định `val_report.json`, chạy `data_trans.py` để xử lý đặc trưng, xuất ra file dữ liệu sẵn sàng huấn luyện `processed_data.parquet`.
    4.  **`calories_train` (Huấn luyện mô hình)**: Kích hoạt `Orchestrator.py` cùng với file tham số `train_params.yaml`. Đầu ra được lưu trữ tĩnh là file trọng số mô hình `model.pkl` tại thư mục Artifacts.
    5.  **`calories_eval_drift` (Đánh giá trôi dữ liệu)**: Bước cuối cùng dùng `eval_drift.py` và cấu hình Evidently để so sánh dữ liệu mới và cũ, kết xuất báo cáo trôi lệch dưới dạng `drift_metrics.json` và `drift_report.html`.

## 10. API Framework ( src/api )

