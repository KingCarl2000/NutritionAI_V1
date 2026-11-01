
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

### D. Danh mục đầy đủ các file hiện có trong `Artifacts`

| File | Vai trò / nội dung |
|---|---|
| `Artifacts/data.dvc` | Chỉ thị DVC theo dõi thư mục dữ liệu `Artifacts/data`, ghi nhận hash MD5, kích thước và số lượng file. |
| `Artifacts/data/ml_calories/processed/calories_features.csv` | Dữ liệu calories đã được tiền xử lý và bổ sung các đặc trưng phục vụ huấn luyện mô hình. |
| `Artifacts/data/ml_calories/processed/train.csv` | Tập huấn luyện của mô hình dự đoán calories, gồm các đặc trưng đầu vào và cột mục tiêu `Calories`. |
| `Artifacts/data/ml_calories/processed/test.csv` | Tập kiểm thử của mô hình dự đoán calories, gồm các đặc trưng đầu vào và cột mục tiêu `Calories`. |
| `Artifacts/data/ml_calories/processed/validation.csv` | Tập validation của mô hình dự đoán calories, gồm các đặc trưng đầu vào và cột mục tiêu `Calories`. |
| `Artifacts/EDA_reports/calories_dataset_schema.html` | Báo cáo hồ sơ dữ liệu được ydata-profiling sinh ra cho tập dữ liệu calories. |
| `Artifacts/EDA_reports/calories_metrics.json` | Các số liệu EDA định lượng của tập dữ liệu calories, dùng cho kiểm tra và giám sát dữ liệu. |
| `Artifacts/reports/ml_calories/evidently/drift_metrics.json` | File kết quả metric phát hiện data drift của Evidently; hiện đang có kích thước 0 byte và cần được sinh lại khi chạy giám sát. |
| `Artifacts/reports/ml_calories/evidently/drift_report.html` | Báo cáo HTML data drift của Evidently; hiện đang có kích thước 0 byte và cần được sinh lại khi chạy giám sát. |
| `Artifacts/reports/ml_calories/validation/val_report.json` | Báo cáo kết quả validation dữ liệu calories; hiện đang có kích thước 0 byte và cần được sinh lại khi chạy pipeline validation. |

> Hiện thư mục `Artifacts` không chứa `quarantine_data.csv`; mục `quarantine_data.csv` ở phần C chỉ mô tả đầu ra có thể được tạo khi phát hiện dữ liệu vi phạm validation, và cần được xác nhận lại trong luồng tạo artifact thực tế.
