# DATA INGESTION PIPELINE SPECIFICATION (`calories_ingest`)

Tài liệu đặc tả kỹ thuật toàn diện cho luồng **Data Ingestion** thuộc hệ thống `NutritionAI_V1`. Tài liệu này được thiết kế tối ưu cho AI/LLM và kỹ sư hệ thống nhằm mục đích:
1. Hiểu rõ luồng dữ liệu từ tầng Pre-Ingestion (PostgreSQL) đến tầng Feature Store, DVC Artifacts và tầng Downstream Validation (`data_val.py`).
2. Nắm bắt chính xác các **hàm con (sub-functions)**, **decorators**, **cấu trúc khóa YAML thực tế**, và **hợp đồng dữ liệu (Data Contract)** để viết mã tích hợp hoặc mở rộng sang các domain mới (ví dụ: `recipes`, `users`).

---

## 1. Định nghĩa DVC Stage & I/O Contract

Luồng `calories_ingest` được quản lý phiên bản và tái lập thông qua DVC (`dvc.yaml`).

### Lệnh khởi tạo DVC Stage
```bash
dvc stage add --force -n calories_ingest \
  -d src/pipelines/ml_pipelines/calories_pred/data_ingest.py \
  -d src/data_io/base_extractor.py \
  -d src/data_io/dvc_handler.py \
  -d src/nutrition_core/logging/logger.py \
  -d src/nutrition_core/logging/mlflow_tracker.py \
  -p config/pipelines/ml_calories/data_ingestion.yaml: \
  -p config/datasets/calories.yaml: \
  -o Artifacts/data/ml_calories/raw \
  -M Artifacts/reports/ml_calories/ingestion/ingestion_report.json \
  python src/pipelines/ml_pipelines/calories_pred/data_ingest.py
```

### Bảng Hợp đồng Đầu vào / Đầu ra (I/O Contract)

| Phân loại DVC | Đường dẫn (Path) | Vai trò & Trạng thái Thực tế |
| :--- | :--- | :--- |
| **Entrypoint** | `src/pipelines/ml_pipelines/calories_pred/data_ingest.py` | Script thực thi chính, chứa class điều phối `CaloriesDataIngestion`. |
| **Dependency (`-d`)** | `src/data_io/base_extractor.py` | Lớp trừu tượng `BasePostgresExtractor` xử lý kết nối DB và ép kiểu DataFrame. |
| **Dependency (`-d`)** | `src/data_io/dvc_handler.py` | Lớp `DVCHandler` quản lý `dvc add`, `dvc pull` và cơ chế fallback cục bộ. |
| **Dependency (`-d`)** | `src/nutrition_core/logging/logger.py` | Cung cấp `logger` đa luồng và các decorator (`@monitor_performance`, `@dvc_error_handler`). |
| **Dependency (`-d`)** | `src/nutrition_core/logging/mlflow_tracker.py` | Lớp `MLOpsTracker` ghi nhận metrics, params, data lineage và system logs lên MLflow. |
| **Params (`-p`)** | `config/pipelines/ml_calories/data_ingestion.yaml` | Cấu hình pipeline: thông tin bảng DB, tỷ lệ chia tập (`data_split`), thư mục xuất (`feature_store`). |
| **Params (`-p`)** | `config/datasets/calories.yaml` | Cấu hình dataset: metadata nguồn `raw.calories`, `target_column`, `columns_to_drop`. |
| **Output (`-o`)** | `Artifacts/data/ml_calories/raw` | Thư mục được DVC theo dõi (cùng `data.dvc` ở root), xuất ra 4 file CSV:<br>• `current_data.csv` (100% dữ liệu nguồn - 15.000 dòng)<br>• `train.csv` (70% - 10.500 dòng)<br>• `val.csv` (15% - 2.250 dòng)<br>• `test.csv` (15% - 2.250 dòng) |
| **Metrics (`-M`)** | `Artifacts/reports/ml_calories/ingestion/ingestion_report.json` | Khai báo DVC metric với **`cache: false`**. Lưu thông số cấu hình, số dòng sau chia tập và trạng thái (`SUCCESS`). |

> **Lưu ý vận hành (Workspace State):** Trong thực tế triển khai, các file trung gian tùy chọn như `docs/architecture/tables_created_report.yaml` hoặc `config/pipelines/ml_calories/calories_data_schema.json` có thể không tồn tại sẵn trong workspace; pipeline `calories_ingest` hoạt động độc lập và lấy nguồn sự thật trực tiếp từ `config/datasets/calories.yaml` và `config/pipelines/ml_calories/data_ingestion.yaml`.

---

## 2. Luồng Dữ Liệu Tổng Thể & Liên Kết Liên Tầng (End-to-End Flow)

Quy trình dữ liệu xung quanh `calories_ingest` gồm 3 giai đoạn nối tiếp:

### Giai đoạn 1: Pre-Ingestion & Profiling (Chạy độc lập qua `src/scripts`)
1. **DDL Migration (`create_tables.py` / Alembic):** Đọc cấu hình YAML để khởi tạo bảng vật lý `raw.calories` trên PostgreSQL.
2. **Bulk ETL (`run_db_bulk_load.py` -> `PostgresBulkLoader`):** Nạp dữ liệu thô từ CSV vào PostgreSQL bằng lệnh `COPY FROM STDIN` tốc độ cao.
3. **EDA Profiling (`eda_report.py`):** Truy vấn mẫu từ PostgreSQL (`LIMIT 10000`) để xuất bộ báo cáo phân tích khám phá tại `Artifacts/EDA_reports/calories_dataset_schema.html` và `Artifacts/EDA_reports/calories_metrics.json`.

### Giai đoạn 2: Core Data Ingestion (`calories_ingest` stage)
1. **Initiate & Load Config:** Khởi tạo `MLOpsTracker`, nạp cấu hình từ `data_ingestion.yaml` và `calories.yaml`.
2. **Extract from PostgreSQL:** Truy vấn bảng `raw.calories` thông qua SQLAlchemy Pool, tự động làm phẳng JSONB (nếu có) và chuẩn hóa kiểu dữ liệu (Pandas/NumPy). Ghi nhận **Data Lineage** lên MLflow.
3. **Drop Columns & Schema Alignment:** Loại bỏ các cột không cần thiết khai báo trong `columns_to_drop`.
4. **Split Train/Val/Test:** Chia tập dữ liệu thành 3 phần độc lập (`train`: 70%, `val`: 15%, `test`: 15%) với `random_state: 42`.
5. **Export & Versioning:** Lưu 4 file `.csv` vào `Artifacts/data/ml_calories/raw`, kích hoạt `dvc add` nếu `dvc_tracked: true`, xuất `ingestion_report.json` (`cache: false`) và đẩy toàn bộ log lên MLflow.

### Giai đoạn 3: Downstream Handshake (`calories_validate` stage)
* Thư mục `Artifacts/data/ml_calories/raw` đóng vai trò là `input_dir` trực tiếp cho bước kiểm định (`src/pipelines/ml_pipelines/calories_pred/data_val.py`).
* Dữ liệu đầu ra của Ingestion phải thỏa mãn ràng buộc lược đồ Pandera (`MlCaloriesInputSchema` và `MlCaloriesTrainingSchema` trong `config/pipelines/ml_calories/calories_schema.py`) và ngưỡng lỗi tối đa `error_threshold: 0.05` (5%) theo `config/pipelines/global.yaml`.

---

## 3. Sơ đồ Gọi Hàm Con & Chi Tiết API (Call Graph & Sub-functions)

AI cần tuân thủ chính xác chuỗi gọi hàm con dưới đây khi phân tích hoặc viết mã mở rộng:

```text
CaloriesDataIngestion.run(report_output_dir)
 ├── MLOpsTracker.start_run(run_name="data_ingestion")
 ├── CaloriesDataIngestion.extract()
 │    └── CaloriesDataIngestion.extract_data_from_postgresql()   [@monitor_performance(time_limit=30)]
 │         └── BasePostgresExtractor.fetch_as_dataframe(...)     [@monitor_performance(time_limit=10.0)]
 │              ├── BasePostgresExtractor._get_connection()      -> src.postgres.core.connection.get_engine()
 │              ├── pd.read_sql(sqlalchemy.text(query), conn)
 │              ├── BasePostgresExtractor._convert_data_types(...)
 │              └── MLOpsTracker.log_dataset_lineage(df, dataset_name, source_uri)
 ├── CaloriesDataIngestion.process_and_split_data(df)            [@monitor_performance()]
 │    └── sklearn.model_selection.train_test_split(...)
 ├── CaloriesDataIngestion.export_to_feature_store(...)          [@monitor_performance(time_limit=15.0)]
 │    ├── pd.DataFrame.to_csv(...)                               -> current_data.csv, train.csv, val.csv, test.csv
 │    └── DVCHandler.add_data(file_path=export_dir)              [@dvc_error_handler]
 ├── MLOpsTracker.log_params_and_metrics(params, metrics)
 ├── CaloriesDataIngestion.export_ingestion_report(report_dir, metrics, params)
 │    └── MLOpsTracker.log_artifact_file(local_path, artifact_path)
 └── MLOpsTracker.end_run(status="FINISHED")
      └── MLOpsTracker.upload_system_logs()
```

### Chi tiết Chữ ký Hàm & Module Thành phần

#### A. Orchestrator Class: `src/pipelines/ml_pipelines/calories_pred/data_ingest.py`
* **`class CaloriesDataIngestion(BasePostgresExtractor)`**
  * `__init__(self, config: dict, tracker: MLOpsTracker = None)`: Khởi tạo `super().__init__()`, `self.dvc = DVCHandler()`, `self.tracker = tracker or MLOpsTracker(...)`, và gọi `self._load_schema()`.
  * `_load_schema(self) -> dict`: Đọc file cấu hình dataset YAML (từ `config['schema']['data_config_file']` hoặc `json_schema_file`) bằng `yaml.safe_load()`.
  * `extract_data_from_postgresql(self) -> pd.DataFrame`: Lấy `schema_name` (`"raw"`) và `table_name` (`"calories"`) từ `config['database']` (fallback về `dataset_schema`), xây dựng câu SQL `SELECT * FROM {schema}.{table}` và gọi `self.fetch_as_dataframe(...)` với `index_col="User_ID"`.
  * `process_and_split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`: Xóa các cột trong `columns_to_drop` (xử lý an toàn kể cả khi `User_ID` đang làm index hoặc cột), chia dữ liệu theo `config['data_split']` (`train_ratio`, `validation_ratio`, `test_ratio`, `random_state=42`).
  * `export_to_feature_store(self, df, train_df, val_df, test_df) -> None`: Ghi 4 file (`current_data.csv`, `train.csv`, `val.csv`, `test.csv`) kèm index vào `config['feature_store']['export_dir']` (`Artifacts/data/ml_calories/raw`). Nếu `dvc_tracked=True`, gọi `self.dvc.add_data(export_dir)`.
  * `export_ingestion_report(self, report_dir: str, metrics: dict, params: dict) -> str`: Xuất file `ingestion_report.json` (chứa tên pipeline, timestamp ISO, params, metrics số dòng và trạng thái `SUCCESS`) và gọi `self.tracker.log_artifact_file()`.
  * `run(self, report_output_dir: str = None) -> dict`: Hàm điều phối chính quản lý vòng đời `try...except...finally` với MLflow.

#### B. Base Extractor: `src/data_io/base_extractor.py`
* **`class BasePostgresExtractor(ABC)`**
  * `_get_connection(self)`: Lấy connection từ pool (`src.postgres.core.connection.get_engine()`), ném `DatabaseConnectionError` nếu thất bại.
  * `_convert_data_types(self, df: pd.DataFrame, schema_mapping: dict = None, flatten_json_cols: list = None) -> pd.DataFrame`:
    * Làm phẳng cột JSONB thành `col_subcol`.
    * Ép kiểu tự động: `Decimal -> float64`, `bool -> pd.BooleanDtype`, `int -> Int64`, `list/array -> np.array`.
    * Ép kiểu tường minh theo `schema_mapping` (hỗ trợ `datetime64[ns]`).
  * `fetch_as_dataframe(self, query: str, params: dict = None, schema_mapping: dict = None, flatten_json_cols: list = None, index_col: str = None, mlflow_tracker: MLOpsTracker = None, dataset_name: str = None) -> pd.DataFrame`: Thực thi SQL, chuẩn hóa kiểu, đặt index và log Data Lineage lên MLflow với URI `nutrition_ai_db_query_{query_hash}`.

#### C. DVC Wrapper: `src/data_io/dvc_handler.py`
* **`class DVCHandler`**
  * `__init__(self, repo: str = None, rev: str = None)`
  * `add_data(self, file_path: str) -> None`: Chạy `dvc add <file_path>` được bọc bởi `@dvc_error_handler` (tự động retry nếu lỗi lock/IO).
  * `sync_data(self) -> None`: Chạy `dvc pull` (tự động retry tối đa 3 lần, nghỉ 2s).
  * `get_dataset(self, data_path: str, format_type: str = 'csv') -> pd.DataFrame`: Đọc dữ liệu từ DVC remote qua `dvc.api.open`, tự động kích hoạt `_fallback_local_dataset` nếu mất kết nối remote.

#### D. Logging & MLOps Tracking: `src/nutrition_core/logging/`
* **`logger.py`**:
  * `logger`: Ghi đồng thời ra Console và 3 luồng file cách ly (`oper_logs`, `warm_logs`, `error_logs`) theo cấu trúc `logs/{dir}/MM_DD_YYYY/HH_MM_SS.log`.
  * `@monitor_performance(time_limit: float = 5.0)`: Decorator đo thời gian thực thi hàm, ghi log `WARNING` nếu vượt ngưỡng `time_limit`.
  * `@dvc_error_handler(max_retries=3, retry_delay=2, fallback_func=None)`: Bắt lỗi DVC và ánh xạ sang `DataNotFoundError`, `VersionNotFoundError`, hoặc `RemoteStorageError`.
* **`mlflow_tracker.py` -> `class MLOpsTracker`**:
  * `start_run(self, run_name: str)`
  * `log_params_and_metrics(self, params: dict, metrics: dict, step: int = None)`
  * `log_dataset_lineage(self, df: pd.DataFrame, dataset_name: str, source_uri: str)`
  * `log_artifact_file(self, local_path: str, artifact_path: str = None)`
  * `log_error_and_tag(self, error_msg: str, tags: dict = None)`
  * `end_run(self, status: str = "FINISHED")`: Tự động gọi `upload_system_logs()` trước khi đóng run.

---

## 4. Cấu trúc File Cấu hình YAML & Hợp đồng Lược đồ Dữ liệu

Khi viết code đọc cấu hình cho pipeline, AI cần ánh xạ đúng các khóa (keys) thực tế của hệ thống:

### 1. Pipeline Config (`config/pipelines/ml_calories/data_ingestion.yaml`)
```yaml
database:
  schema_name: "raw"       # Không gian lưu trữ trong PostgreSQL
  table_name: "calories"   # Tên bảng nguồn trong PostgreSQL

schema:
  data_config_file: "config/datasets/calories.yaml"
  json_schema_file: "config/pipelines/ml_calories/calories_data_schema.json"
  target_column: "Calories"
  columns_to_drop: []      # Danh sách cột loại bỏ trước khi chia tập (nếu có)

data_split:
  train_ratio: 0.70
  validation_ratio: 0.15
  test_ratio: 0.15
  random_state: 42

feature_store:
  export_dir: "Artifacts/data/ml_calories/raw"
  dvc_tracked: true

reporting:
  report_dir: "Artifacts/reports/ml_calories/ingestion"
```

### 2. Dataset Config (`config/datasets/calories.yaml`)
```yaml
dataset_name: "calories"
database:
  schema_name: "raw"
  table_name: "calories"
schema:
  index_col: "User_ID"
  target_column: "Calories"
  columns_to_drop:
    - "created_at"
    - "updated_at"
```

### 3. Hợp đồng Cột & Ràng buộc Dữ liệu Đầu ra (Downstream Schema Contract)
Các file CSV xuất ra tại `Artifacts/data/ml_calories/raw/*.csv` sẽ được kiểm định ngay sau đó bởi `MlCaloriesTrainingSchema` (`config/pipelines/ml_calories/calories_schema.py`). AI cần đảm bảo quá trình ép kiểu trong Ingestion không làm sai lệch cấu trúc 9 trường sau:

| Tên cột | Kiểu dữ liệu chuẩn | Ràng buộc nghiệp vụ & sinh học (Pandera) | Vai trò |
| :--- | :--- | :--- | :--- |
| `User_ID` | `Int64` (Index/Col) | Số nguyên dương (`>= 1`) | Định danh bản ghi |
| `Gender` | `string` / `category` | Giá trị thuộc `["male", "female", "other"]` | Categorical Feature |
| `Age` | `Int64` / `float64` | Trong khoảng `[0, 200]` tuổi | Numerical Feature |
| `Height` | `float64` | Trong khoảng `[40.0, 255.0]` cm | Numerical Feature |
| `Weight` | `float64` | Trong khoảng `[2.0, 300.0]` kg | Numerical Feature |
| `Duration` | `float64` | Trong khoảng `[0.0, 300.0]` phút | Numerical Feature |
| `Heart_Rate` | `float64` | Trong khoảng `[40.0, 220.0]` bpm | Numerical Feature |
| `Body_Temp` | `float64` | Trong khoảng `[35.0, 42.5]` °C | Numerical Feature |
| `Calories` | `float64` | Trong khoảng `[1.0, 5000.0]` kcal | **Target Column** |

> **Ràng buộc SLA Toàn cục (`config/pipelines/global.yaml`):**
> * Thời gian thực thi các hàm kiểm định/ép kiểu dữ liệu không được vượt quá **10%** tổng thời gian quy trình ETL Batch (`validation_sla`).
> * Tỷ lệ vi phạm dữ liệu tối đa ở bước kế tiếp là **5%** (`data_failure_rate`: `< 1%` auto-drop, `1% - 5%` quarantine + warning, `> 5%` fail-fast).

---

## 5. Hướng dẫn AI Mở rộng cho Pipeline Mới (Integration Blueprint)

Khi được yêu cầu tạo một pipeline ingestion mới (ví dụ: `recipes_ingest`), AI **bắt buộc** thực hiện theo 4 bước chuẩn sau để duy trì tính đồng nhất của toàn hệ thống:

1. **Tạo File Cấu hình Chuẩn Khóa:**
   * Tạo `config/datasets/<domain>.yaml` và `config/pipelines/ml_<domain>/data_ingestion.yaml` sử dụng đúng cấu trúc các block: `database`, `schema`, `data_split`, `feature_store`.
2. **Kế thừa `BasePostgresExtractor`:**
   * Tạo `src/pipelines/ml_pipelines/<domain>_pred/data_ingest.py`.
   * Khởi tạo class `<Domain>DataIngestion(BasePostgresExtractor)`.
   * Tái sử dụng trực tiếp `self.fetch_as_dataframe()`, `self.dvc.add_data()`, và `self.tracker` thay vì viết lại logic kết nối PostgreSQL hoặc gọi DVC CLI thủ công.
   * Đảm bảo xuất đủ 4 file (`current_data.csv`, `train.csv`, `val.csv`, `test.csv`) vào `feature_store.export_dir` để tương thích với luồng `data_val.py` và `eval_drift.py`.
3. **Áp dụng Decorators Giám sát & Xử lý Lỗi:**
   * Gắn `@monitor_performance(time_limit=30)` cho hàm trích xuất DB và `@monitor_performance(time_limit=15)` cho hàm xuất file.
   * Bọc hàm `run()` trong khối `try ... except Exception as e:` và gọi `self.tracker.log_error_and_tag(str(e))` trước khi `raise` ngoại lệ.
4. **Đăng ký vào `src/scripts/setup_dvc_stages.sh`:**
   * Thêm block `dvc stage add --force -n <domain>_ingest ...` trỏ đúng vào các file dependency lõi (`base_extractor.py`, `dvc_handler.py`, `logger.py`, `mlflow_tracker.py`).
   * Khai báo `-o Artifacts/data/ml_<domain>/raw` và `-M Artifacts/reports/ml_<domain>/ingestion/ingestion_report.json` (lưu ý cấu hình `cache: false` cho file metric JSON trong `dvc.yaml`).