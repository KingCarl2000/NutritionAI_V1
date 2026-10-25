
# DEVELOPER GUIDE: NutritionAI_V1 Config Modules

Tài liệu này tổng hợp danh sách các class và cấu hình schema thuộc phân hệ config, giúp nhà phát triển dễ dàng tra cứu ràng buộc dữ liệu đầu vào cho quá trình huấn luyện và dự đoán.

## 1. Data Validation (Hệ sinh thái Pandera)

*File:* config/pipelines/ml_calories/calories_schema.py

- *MlCaloriesInputSchema* *(pa.DataFrameModel):* Lớp định nghĩa schema cho dữ liệu đầu vào. Tích hợp lớp `Config` để tự động ép kiểu (`coerce = True`) và kiểm soát nghiêm ngặt các cột (`strict = True`). Định nghĩa kiểu dữ liệu và giới hạn giá trị (min/max) cho các trường: `User_ID`, `Gender`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`.
- *MlCaloriesTrainingSchema* *(MlCaloriesInputSchema):* Class schema dành riêng cho tiến trình huấn luyện (Training Pipeline), kế thừa toàn bộ cấu trúc từ `MlCaloriesInputSchema` và bổ sung thêm cột Target là `Calories` (ràng buộc giá trị hợp lệ từ 1.0 đến 314.0).
