#!/bin/bash
set -e # Dừng script ngay khi có bất kỳ lệnh nào bị lỗi

# Di chuyển về thư mục gốc của dự án (NutritionAI_V1)
cd "$(dirname "$0")/../.." || exit

echo "🚀 Bắt đầu cấu hình DVC Pipelines cho phân hệ Calories Prediction..."

# 1. Stage: Data Ingestion (Trích xuất dữ liệu)
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

# 2. Stage: Data Preprocessing (Bổ sung tiếp tại đây)
# dvc stage add --force -n calories_preprocess ...