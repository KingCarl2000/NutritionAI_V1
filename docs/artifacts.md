
## Hệ thống Artifacts & Reports (Các file sinh ra tự động)

Trong quá trình vận hành luồng MLOps và ETL, hệ thống tự động kết xuất ra các tài nguyên báo cáo, hồ sơ dữ liệu (profile) và dữ liệu cách ly. Dưới đây là sơ đồ quản lý các file sinh ra tự động:

### A. Phân tích Khám phá & Hồ sơ dữ liệu (Data Profiling)

* **`Artifacts/EDA_reports/calories_metrics.json`** & **`calories_dataset_schema.html`**
  * *Nguồn gốc:* Khởi chạy từ `src/scripts/eda_report.py`.
  * *Ý nghĩa:* Báo cáo Exploratory Data Analysis (EDA). File `.html` cung cấp biểu đồ trực quan cho con người, còn file `.json` lưu các số liệu định lượng (min, max, missing rate) phục vụ hệ thống máy móc theo dõi.
* **`config/pipelines/ml_calories/calories_data_schema.json`**
  * *Nguồn gốc:* Khởi chạy từ `src/scripts/generate_data_schema.py` kết hợp điều phối bởi `src/scripts/create_schema_pipelines.py`.
  * *Ý nghĩa:* Đóng vai trò là "Data Profile", ghi nhận cấu trúc thực tế và các phân phối của dataset tại thời điểm lấy mẫu để làm Baseline (Reference Data) cho việc phát hiện Data Drift sau này.

### B. Hạ tầng Database (Infrastructure & ETL)

* **`docs/architecture/tables_created_report.yaml`**
  * *Nguồn gốc:* Khởi chạy từ `src/scripts/create_tables.py`.
  * *Ý nghĩa:* Log xác nhận tự động danh sách các bảng (tables) cùng cấu trúc tương ứng đã được ánh xạ và khởi tạo thành công vào CSDL PostgreSQL.

### C. Giai đoạn Kiểm định & Giám sát (Validation & Monitoring)

* **`Artifacts/reports/ml_calories/validation/quarantine_data.csv`** & **`val_report.json`**
  * *Nguồn gốc:* Quá trình thực thi `src/pipelines/ml_pipelines/calories_pred/data_val.py`.
  * *Ý nghĩa:* Vùng cách ly (Quarantine). Chứa các bản ghi vi phạm giới hạn sinh lý thực tế được định nghĩa bằng Pandera và báo cáo chi tiết về nguyên nhân thất bại của từng dòng.
* **`Artifacts/reports/ml_calories/evidently/drift_metrics.json`** & **`drift_report.html`**
  * *Nguồn gốc:* Sinh ra từ các tác vụ thuộc nhóm Evidently AI.
  * *Ý nghĩa:* Báo cáo so sánh sự sai lệch phân phối (Data Drift) giữa dữ liệu nạp mới và dữ liệu tham chiếu gốc, giúp kích hoạt các cơ chế huấn luyện lại mô hình (retrain) nếu cần thiết.
