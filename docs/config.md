# DEVELOPER GUIDE: NutritionAI_V1 Config Modules

Tài liệu này tổng hợp danh sách các class và cấu hình schema thuộc phân hệ config, giúp nhà phát triển dễ dàng tra cứu ràng buộc dữ liệu đầu vào cho quá trình huấn luyện và dự đoán.

## 1. Cấu hình Lược đồ Kiểm định Dữ liệu (Data Validation Schema)

*File:* [config/pipelines/ml_calories/calories_schema.py](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/config/pipelines/ml_calories/calories_schema.py)

- **Mục đích**: Định nghĩa lược đồ dữ liệu (schema) chặt chẽ bằng Pandera để kiểm định dữ liệu đầu vào và dữ liệu huấn luyện, dựa trên các ngưỡng sinh học thực tế.
- **Thành phần chính**:
    - **`MlCaloriesInputSchema` (pa.DataFrameModel)**: Lớp kiểm định dữ liệu đầu vào gốc.
        - Tích hợp lớp `Config` để tự động ép kiểu (`coerce = True`) và kiểm soát nghiêm ngặt các cột, từ chối cột không xác định (`strict = True`).
        - **Quy tắc (Sinh học & Logic)**:
            - `User_ID`: Số nguyên dương (`>= 1`).
            - `Gender`: Giới hạn trong `["male", "female", "other"]`.
            - `Age`: Từ `0` đến `200` tuổi.
            - `Height`: Chiều cao từ `40` đến `255.0` cm.
            - `Weight`: Cân nặng từ `2` đến `300.0` kg.
            - `Duration`: Thời gian tập từ `0` đến `300.0` phút.
            - `Heart_Rate`: Nhịp tim từ `40.0` đến `220.0` bpm.
            - `Body_Temp`: Nhiệt độ cơ thể từ `35.0` °C đến `42.5` °C.
    - **`MlCaloriesTrainingSchema`**: Lớp schema dành riêng cho tiến trình huấn luyện (Training Pipeline).
        - Kế thừa toàn bộ đặc tính từ `MlCaloriesInputSchema`.
        - Bổ sung kiểm định cột Target **`Calories`**: Ràng buộc giá trị calo tiêu thụ hợp lệ từ `1.0` đến `5000.0` kcal (bao phủ cả các phiên tập luyện cực nặng).

## 2. Cấu hình Pipeline Kiểm định Dữ liệu (Data Validation)

*File:* [config/pipelines/ml_calories/data_val.yaml](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/config/pipelines/ml_calories/data_val.yaml)

- **Mục đích**: Hoạt động như một bảng tra cứu API (API Registry), khai báo đường dẫn pipeline, thư mục lưu trữ đồng bộ DVC và thiết lập tham số ngưỡng lỗi cho quy trình kiểm định dữ liệu (Data Validation).
- **Thành phần chính**:
    - **Tham chiếu API Cốt lõi (`core_api_references`)**: 
        - Quản lý đường dẫn import động tới các thành phần cơ sở: lớp Schema chuẩn (`BaseDataFrameModel`), các hàm kiểm tra tùy chỉnh (`is_in_numeric_range`, `is_valid_string_length`).
        - Xử lý ngoại lệ (`DataValidationError`) và theo dõi/ghi log tự động qua decorator (`validation_error_handler`, `MLOpsTracker`).
    - **Lớp Lược đồ (`Schema Classes`)**: Chỉ định linh hoạt các lớp kiểm định áp dụng cho từng giai đoạn, bao gồm `train_schema` (cho huấn luyện) và `inference_schema` (cho suy luận thực tế).
    - **Đường dẫn & Quản lý Thư mục DVC (`pipeline_paths`)**:
        - Trỏ tới các file cấu hình liên quan như `calories_schema.py` và `evidently_profile.yaml`.
        - **Đồng bộ DVC**: Khai báo rõ các thư mục đầu vào (`input_dir`), đầu ra sau kiểm định (`output_dir`) và thư mục cách ly dữ liệu hỏng (`quarantine_dir`) đặt tại `Artifacts/data/ml_calories/`.
    - **Tham số Kiểm định (`validation_params`)**: Đặt ngưỡng lỗi tối đa `error_threshold: 0.05` (tương đương 5%). Nếu tỷ lệ dữ liệu vi phạm vượt ngưỡng này, hệ thống sẽ kích hoạt cơ chế cảnh báo hoặc ngắt khẩn cấp.

## 3. Cấu hình Giám sát Mô hình & Data Drift (Evidently AI Profile)

*File:* [config/pipelines/ml_calories/evidently_profile.yaml](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/config/pipelines/ml_calories/evidently_profile.yaml)

- **Mục đích**: Khai báo cấu hình cho hệ sinh thái Evidently AI để theo dõi hiện tượng trôi lệch dữ liệu (data drift), định nghĩa mapping các đặc trưng, và thiết lập các phép kiểm định thống kê (statistical tests) tự động.
- **Thành phần chính**:
    - **Cấu hình Engine (`engine_config`)**: Thiết lập tên project (`Calories_Prediction_Drift_Monitoring`) và thư mục lưu trữ báo cáo HTML/JSON tại `Artifacts/reports/ml_calories/evidently`.
    - **Ánh xạ Đặc trưng (`column_mapping`)**: 
        - Xác định vai trò của các cột quan trọng: `target` ("calories"), `datetime` ("created_at").
        - Phân nhóm biến cụ thể: `numerical_features` (age, weight, height, duration, heart_rate, body_temp) và `categorical_features` (gender) để áp dụng thuật toán kiểm định phù hợp.
    - **Nguồn Dữ liệu Đánh giá (`current_data`)**: Kết nối trực tiếp đến bảng `calories` trong không gian `raw` của cơ sở dữ liệu `postgresql`.
    - **Ngưỡng và Phép kiểm định Data Drift (`data_drift`)**:
        - **Ngưỡng toàn cục (`dataset_drift_share`)**: Đánh giá toàn bộ tập dữ liệu bị trôi lệch nếu 50% (0.5) số lượng cột có sự thay đổi.
        - **Phép kiểm định mặc định**: Sử dụng `wasserstein` (ngưỡng 0.1) cho dữ liệu số và `jensenshannon` (ngưỡng 0.1) cho dữ liệu phân loại.
        - **Ghi đè đặc thù (`per_column_options`)**: Cấu hình kiểm tra khắt khe hơn cho từng biến cụ thể. Ví dụ: biến `age` dùng kiểm định Kolmogorov-Smirnov (`ks`) với ngưỡng 0.05, và biến `gender` dùng Population Stability Index (`psi`) với ngưỡng 0.1.

## 4. Cấu hình Nạp Dữ liệu & Nguồn Dataset (Data Ingestion)

*Files:* 
- [`config/datasets/calories.yaml`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/config/datasets/calories.yaml)
- [`config/pipelines/ml_calories/data_ingestion.yaml`](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/config/pipelines/ml_calories/data_ingestion.yaml)

- **Mục đích**: Khai báo tham số đầu vào, siêu dữ liệu (metadata) và định tuyến đường ống (pipeline) phục vụ cho bước thu thập, tiền xử lý cơ bản và phân chia tập dữ liệu.
- **Thành phần chính**:
    - **Thông tin cơ sở dữ liệu (`database`)**: Chỉ định định danh không gian lưu trữ (`schema_name: "raw"`) và tên bảng cần trích xuất (`table_name: "calories"`).
    - **Cấu hình Lược đồ & Xử lý cột (`schema`)**: 
        - Liên kết đồng bộ JSON schema: `json_schema_file` (để đồng bộ với Database Models).
        - Khai báo rõ cột mục tiêu: `target_column: "Calories"`.
        - Chỉ định danh sách các cột cần loại bỏ trước khi đưa vào huấn luyện qua `columns_to_drop` (VD: loại bỏ `"User_ID"`).
    - **Phân chia Dữ liệu (`data_split`)**: Cấu hình tỷ lệ chia tách tiêu chuẩn đảm bảo khả năng tái lập (reproducible qua `random_state: 42`):
        - Train (`train_ratio`): 70%
        - Test (`test_ratio`): 15%
        - Validation (`validation_ratio`): 15%
    - **Quản lý Feature Store (`feature_store`)**: 
        - Thư mục xuất dữ liệu: `export_dir` chỉ định xuất các tập dataframe thô ra `Artifacts/data/ml_calories/raw`.
        - Quản lý phiên bản: Kích hoạt theo dõi file bằng DVC (`dvc_tracked: true`).

## 5. Cấu hình Ngưỡng MLOps Toàn cục (Global Thresholds)

*File:* [config/pipelines/global.yaml](https://github.com/KingCarl2000/NutritionAI_V1/blob/main/config/pipelines/global.yaml)

- **Mục đích**: Định nghĩa tập trung toàn bộ các cấu hình, ngưỡng cảnh báo (thresholds) và SLA dùng chung cho mọi pipelines trong hệ thống MLOps.
- **Thành phần chính**:
    - **Tỷ lệ Vi phạm Dữ liệu (`data_failure_rate`)**: Quản lý dữ liệu lỗi ở bước validation. Tự động loại bỏ dòng hỏng nếu lỗi < 1% (`info_threshold`), đưa vào Quarantine ghi log cảnh báo từ 1% - 5% (`warning_threshold`), và ngắt khẩn cấp pipeline (Fail-Fast) nếu vượt 5%.
    - **Trôi lệch Phân phối & Thống kê (`data_drift`)**: Quản lý hiện tượng trôi lệch dữ liệu tích hợp với Evidently AI. 
        - Đặt ngưỡng PSI chung: cảnh báo khi `> 0.10` và yêu cầu Retrain khi `> 0.25`.
        - Cấu hình kiểm định chi tiết theo loại dữ liệu: Numeric (K-S test, Wasserstein, t-test), Categorical (Chi-square, Z-test, Jensen-Shannon), Text, và Embeddings (Cosine, ROC-AUC, MMD). Đặt `p_value_alpha` ở mức ý nghĩa `0.05`.
    - **Độ trễ Dịch vụ & SLA (`validation_sla` & `service_latency`)**: Ràng buộc hiệu năng xử lý. Yêu cầu độ trễ trung vị P50 < 50ms và P95 < 200ms cho Online API; thời gian kiểm định không vượt quá 10% tổng thời gian quy trình ETL Batch.
    - **Trạng thái Khả dụng (`service_availability`)**: Tần suất kiểm tra sức khỏe `60s`/lần, yêu cầu 3 lần fail liên tiếp trước khi báo lỗi (tránh flapping).
    - **Tỷ lệ Lỗi API (`api_error_rate`)**: Cảnh báo khi lỗi HTTP 5xx vượt 1% và đạt mức nghiêm trọng khi vượt 5% (yêu cầu tối thiểu 10 requests để đánh giá).
    - **Bão hòa Tài nguyên (`resource_saturation`)**: Cảnh báo khi ổ đĩa hoặc RAM đạt 85% dung lượng thực tế. Dự báo xu hướng cạn kiệt tài nguyên trong 4 giờ tới (`prediction_horizon_hours`).
    - **Chất lượng Dữ liệu & Mô hình (`data_quality_and_model`)**: Cấu hình kiểm soát chất lượng như: phát hiện cột gần như hằng số (95% trùng lặp), độ tin cậy thống kê mặc định (95%), và ngưỡng xác suất phân loại (0.5).
