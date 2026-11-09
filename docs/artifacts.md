## Artifacts và reports

Tài liệu này mô tả các artifact được tạo bởi pipeline calories, cùng trạng thái quan sát được trong workspace ngày 09/11/2026. Các đường dẫn cấu hình đầu ra được phân biệt với những file đã thực sự tồn tại: pipeline chỉ tạo các file đó sau khi chạy thành công đến bước tương ứng.

### 1. Nạp dữ liệu (ingestion)

Pipeline `calories_ingest` trong `dvc.yaml` chạy `src/pipelines/ml_pipelines/calories_pred/data_ingest.py`.

| Đường dẫn | Ý nghĩa | Trạng thái hiện tại |
|---|---|---|
| `Artifacts/data/ml_calories/raw/current_data.csv` | Dữ liệu nguồn hiện hành được xuất trong bước ingestion. | Có trong workspace. |
| `Artifacts/data/ml_calories/raw/train.csv` | Phần train; tỷ lệ cấu hình 70%. | Có trong workspace. |
| `Artifacts/data/ml_calories/raw/val.csv` | Phần validation; tỷ lệ cấu hình 15%. | Có trong workspace. |
| `Artifacts/data/ml_calories/raw/test.csv` | Phần test; tỷ lệ cấu hình 15%. | Có trong workspace. |
| `Artifacts/reports/ml_calories/ingestion/ingestion_report.json` | Thông số, số dòng sau khi chia tập và trạng thái ingestion. Được DVC khai báo là metric với `cache: false`. | Có trong workspace; báo cáo ghi trạng thái `SUCCESS`, tổng 15.000 dòng, train 10.500, validation 2.250, test 2.250. |

Các tỷ lệ chia và `random_state` được cấu hình trong `config/pipelines/ml_calories/data_ingestion.yaml`. DVC stage theo dõi thư mục `Artifacts/data/ml_calories/raw` và báo cáo ingestion.

### 2. Báo cáo EDA

| Đường dẫn | Ý nghĩa | Trạng thái hiện tại |
|---|---|---|
| `Artifacts/EDA_reports/calories_dataset_schema.html` | Báo cáo hồ sơ dữ liệu trực quan do `src/scripts/eda_report.py` tạo. | Có trong workspace. |
| `Artifacts/EDA_reports/calories_metrics.json` | Các thống kê EDA định lượng của dữ liệu calories. | Có trong workspace. |

### 3. Kiểm định dữ liệu (validation)

`config/pipelines/ml_calories/data_val.yaml` đặt thư mục đầu vào là `Artifacts/data/ml_calories/raw`, thư mục dữ liệu hợp lệ là `Artifacts/data/ml_calories/validated`, thư mục cách ly là `Artifacts/data/ml_calories/quarantine`, và ngưỡng lỗi là 5%. Theo cập nhật trong `src/pipelines/ml_pipelines/calories_pred/data_val.py`, bước validation ghi báo cáo theo mode, lưu dữ liệu hợp lệ và khi tỷ lệ lỗi vượt ngưỡng sẽ lưu dữ liệu đầu vào vào quarantine trước khi dừng pipeline.

| Đường dẫn mẫu | Ý nghĩa |
|---|---|
| `Artifacts/data/ml_calories/validated/validation_report_<mode>.json` | Báo cáo mode, tổng số dòng, số lỗi, tỷ lệ lỗi và tối đa 100 chi tiết lỗi đầu tiên. |
| `Artifacts/data/ml_calories/validated/valid_<mode>.csv` | Dữ liệu hợp lệ được ghi sau khi validation hoàn tất trong ngưỡng. |
| `Artifacts/data/ml_calories/quarantine/quarantine_<mode>.csv` | Dữ liệu đầu vào được cách ly nếu tỷ lệ lỗi vượt ngưỡng. |

`<mode>` hiện được thiết kế cho `train` hoặc `inference`. Các đường dẫn này là đầu ra cấu hình/dự kiến; hiện tại workspace chưa có thư mục `validated` hay `quarantine`, vì vậy chưa có báo cáo validation hoặc file quarantine được tạo. File quarantine chỉ được ghi khi tỷ lệ lỗi vượt ngưỡng.

### 4. Giám sát drift

Các tác vụ Evidently ghi báo cáo vào `Artifacts/reports/ml_calories/evidently`:

| Đường dẫn | Ý nghĩa | Trạng thái hiện tại |
|---|---|---|
| `Artifacts/reports/ml_calories/evidently/drift_metrics.json` | Các metric phát hiện data drift. | Chưa có trong workspace. |
| `Artifacts/reports/ml_calories/evidently/drift_report.html` | Báo cáo drift dạng HTML. | Chưa có trong workspace. |

### 5. Danh mục artifact hiện có

Danh sách này phản ánh các file đang có dưới `Artifacts` trong workspace; `data.dvc` nằm ở thư mục gốc repository, không nằm trong `Artifacts`.

| Đường dẫn | Vai trò / nội dung |
|---|---|
| `Artifacts/data/ml_calories/raw/current_data.csv` | Dữ liệu nguồn hiện hành. |
| `Artifacts/data/ml_calories/raw/train.csv` | Tập train được xuất bởi ingestion. |
| `Artifacts/data/ml_calories/raw/val.csv` | Tập validation được xuất bởi ingestion. |
| `Artifacts/data/ml_calories/raw/test.csv` | Tập test được xuất bởi ingestion. |
| `Artifacts/EDA_reports/calories_dataset_schema.html` | Báo cáo EDA HTML. |
| `Artifacts/EDA_reports/calories_metrics.json` | Metric EDA dạng JSON. |
| `Artifacts/reports/ml_calories/ingestion/ingestion_report.json` | Báo cáo ingestion; lần ghi hiện tại báo thành công với 15.000 dòng. |

Các file thuộc thư mục validation và Evidently trong những phần trên là đầu ra theo cấu hình, chưa phải artifact hiện có. Báo cáo tạo bảng tại `docs/architecture/tables_created_report.yaml` và data profile tại `config/pipelines/ml_calories/calories_data_schema.json` không xuất hiện trong danh sách `Artifacts` này và hiện không được xác nhận là có trong workspace.
