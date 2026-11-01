# DEVELOPER GUIDE: NutritionAI_V1 Config Modules

Tài liệu này tổng hợp danh sách các class và cấu hình schema thuộc phân hệ config, giúp nhà phát triển dễ dàng tra cứu ràng buộc dữ liệu đầu vào cho quá trình huấn luyện và dự đoán.

## 1. Data Validation (Hệ sinh thái Pandera)

*File:* config/pipelines/ml_calories/calories_schema.py

- *MlCaloriesInputSchema* *(pa.DataFrameModel):* Lớp định nghĩa schema cho dữ liệu đầu vào. Tích hợp lớp `Config` để tự động ép kiểu (`coerce = True`) và kiểm soát nghiêm ngặt các cột (`strict = True`). Định nghĩa kiểu dữ liệu và giới hạn giá trị (min/max) cho các trường: `User_ID`, `Gender`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`.
- *MlCaloriesTrainingSchema* *(MlCaloriesInputSchema):* Class schema dành riêng cho tiến trình huấn luyện (Training Pipeline), kế thừa toàn bộ cấu trúc từ `MlCaloriesInputSchema` và bổ sung thêm cột Target là `Calories` (ràng buộc giá trị hợp lệ từ 1.0 đến 314.0).


## 2. API Registry cho Data Validation (Nutrition Pandera Core)

*File: [config/pipelines/ml_calories/data_val.yaml](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/config/pipelines/ml_calories/data_val.yaml)*

- **Mục đích**: Bảng tra cứu API (API Registry) để gọi nhanh các hàm, lớp cơ sở dùng chung cho pipeline kiểm định.
- **Thành phần chính**:
  - `base_schema` & `validate_df`: Liên kết tới lớp cơ sở (`BaseDataFrameModel`) quản lý cấu hình `strict`, `coerce` và kiểm định khung toàn dự án.
  - `custom_checks`: Ánh xạ các quy tắc kiểm tra tùy chỉnh như khoảng số sinh lý (`numeric_range_check`) và độ dài chuỗi (`string_length_check`).
  - `exceptions` & `validation_handler`: Định tuyến các ngoại lệ vi phạm Schema và bộ decorator bắt lỗi, cách ly dữ liệu hỏng (Quarantine).

## 3. Cấu hình Giám sát Mô hình (Evidently AI Profile)

*File: [config/pipelines/ml_calories/evidently_profile.yaml](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/config/pipelines/ml_calories/evidently_profile.yaml)*

- **Mục đích**: Khai báo cấu hình cho hệ sinh thái Evidently AI để theo dõi trôi lệch dữ liệu (data drift) và đánh giá chất lượng pipeline.
- **Thành phần chính**:
  - `data_format`: Định dạng file đầu vào (phân tách bằng dấu phẩy `,`, có header).
  - `column_mapping`: Ánh xạ vai trò các cột, bao gồm `target` ("Calories"), `prediction`, cột ID ("User_ID"). Phân nhóm cụ thể cho biến dạng số (`numerical_features` như Age, Height, Weight) và biến phân loại (`categorical_features`).

## 4. Cấu hình Nạp Dữ liệu & Nguồn Dataset (Data Ingestion)

*File: [config/pipelines/ml_calories/data_ingestion.yaml](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/config/pipelines/ml_calories/data_ingestion.yaml)* & *[config/datasets/calories.yaml](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/config/datasets/calories.yaml)*

- **Mục đích**: Khai báo các tham số đầu vào và siêu dữ liệu (metadata) phục vụ cho bước thu thập và nạp dữ liệu (Data Ingestion).
- **Thành phần chính**:
  - Chỉ định `db_schema` (ví dụ: raw) và tên bảng (`calories`) trong cơ sở dữ liệu PostgreSQL.
  - Đường dẫn ánh xạ cục bộ (`source_path`) tới tệp CSV (VD: `D:/NutritionAI_V1/Data/raw/calories.csv`).

## 5. Cấu hình Ngưỡng MLOps Toàn cục (Global Thresholds)

*File: [config/pipelines/global.yaml](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/config/pipelines/global.yaml)*

- **Mục đích**: Định nghĩa tập trung toàn bộ các cấu hình, ngưỡng cảnh báo (thresholds) và SLA dùng chung cho mọi pipelines trong hệ thống MLOps.
- **Thành phần chính**:
  - `data_failure_rate`: Ngưỡng vi phạm dữ liệu cho phép (cảnh báo hoặc ngắt khẩn cấp Fail-Fast khi vượt 5%).
  - `data_drift`: Ngưỡng cho các kiểm định thống kê trôi lệch chuyên sâu tích hợp Evidently AI (PSI, test p-value như K-S test, Z-test, Wasserstein distance).
  - `validation_sla` & `service_latency`: Ngưỡng độ trễ xử lý (P50, P95 tối đa 200ms) và giới hạn thời lượng batch.
  - `service_availability` & `api_error_rate`: Tần suất Healthcheck (60s), giới hạn lỗi 5xx của API.
  - `resource_saturation`: Cảnh báo mức độ tiêu thụ RAM/Disk thực tế (> 85%), kèm dự báo cạn kiệt tài nguyên.
