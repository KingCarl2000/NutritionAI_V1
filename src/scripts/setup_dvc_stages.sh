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

