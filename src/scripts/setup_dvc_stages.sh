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

# 2. Stage: Data validation & data drift detection (Kiểm tra dữ liệu)
dvc stage add --force -n calories_validate \
  -d src/pipelines/ml_pipelines/calories_pred/data_val.py \
  -d Artifacts/data/ml_calories/raw/train.csv \
  -d src/pipelines/ml_pipelines/calories_pred/eval_drift.py \
  -d src/nutrition_core/pandera/api/base.py \
  -d src/nutrition_core/pandera/api/check.py \
  -d src/nutrition_core/pandera/backends/pandas_executor.py \
  -d src/nutrition_core/pandera/config.py \
  -d src/nutrition_core/pandera/decorators.py \
  -d src/nutrition_core/pandera/evidently_engine.py \
  -d src/data_io/dvc_handler.py \
  -d src/nutrition_core/logging/logger.py \
  -d src/nutrition_core/logging/mlflow_tracker.py \
  -d config/pipelines/ml_calories/calories_schema.py \
  -p config/pipelines/ml_calories/data_val.yaml \
  -p config/pipelines/ml_calories/evidently_profile.yaml \
  -p config/pipelines/global.yaml \
  -o Artifacts/data/ml_calories/validated \
  -M Artifacts/reports/ml_calories/validation/validation_report.json \
  python src/pipelines/ml_pipelines/calories_pred/data_val.py

# dvc stage add --force -n calories_preprocess ...