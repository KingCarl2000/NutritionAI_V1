#!/bin/bash

# Di chuyển về thư mục gốc của dự án (NutritionAI_V1) để DVC nhận diện đúng đường dẫn tương đối
cd "$(dirname "$0")/../.." || exit

echo "🚀 Bắt đầu cấu hình DVC Pipelines cho phân hệ Calories Prediction..."

# 1. Stage: Data Ingestion (Trích xuất dữ liệu)
dvc stage add -n calories_ingest -f \
  -d src/pipelines/ml_pipelines/calories_pred/data_ingest.py \
  -d config/pipelines/ml_calories/data_ingestion.yaml \
  -o Artifacts/reports/ml_calories/ingestion/calories_data_schema.json \
  -o Artifacts/data/ml_calories/raw \
  python src/pipelines/ml_pipelines/calories_pred/data_ingest.py

# 2. Stage: Data Validation (Kiểm định chất lượng với Pandera)
dvc stage add -n calories_validate -f \
  -d src/pipelines/ml_pipelines/calories_pred/data_val.py \
  -d config/pipelines/ml_calories/data_val.yaml \
  -d config/pipelines/ml_calories/calories_schema.py \
  -d Artifacts/data/ml_calories/raw \
  -o Artifacts/data/ml_calories/validated \
  -o Artifacts/data/ml_calories/quarantine \
  python src/pipelines/ml_pipelines/calories_pred/data_val.py

# 3. Stage: Data Transformation (Làm sạch & Biến đổi dữ liệu)
dvc stage add -n calories_transform -f \
  -d src/pipelines/ml_pipelines/calories_pred/data_trans.py \
  -d Artifacts/reports/ml_calories/validation/val_report.json \
  -o Artifacts/data/ml_calories/processed_data.parquet \
  python src/pipelines/ml_pipelines/calories_pred/data_trans.py

# 4. Stage: Model Training (Huấn luyện mô hình - sử dụng Orchestrator.py)
dvc stage add -n calories_train -f \
  -d src/pipelines/ml_pipelines/calories_pred/Orchestrator.py \
  -d config/pipelines/ml_calories/train_params.yaml \
  -d Artifacts/data/ml_calories/processed_data.parquet \
  -o Artifacts/models/ml_calories/model.pkl \
  python src/pipelines/ml_pipelines/calories_pred/Orchestrator.py

# 5. Stage: Data Drift Evaluation (Đánh giá độ trôi dữ liệu với Evidently AI)
dvc stage add -n calories_eval_drift -f \
  -d src/pipelines/ml_pipelines/calories_pred/eval_drift.py \
  -d config/pipelines/ml_calories/evidently_profile.yaml \
  -o Artifacts/reports/ml_calories/evidently/drift_metrics.json \
  -o Artifacts/reports/ml_calories/evidently/drift_report.html \
  python src/pipelines/ml_pipelines/calories_pred/eval_drift.py

echo "✅ Đã tạo xong các stages trong file dvc.yaml!"