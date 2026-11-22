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

# 2. Stage: Data Validation & Data Drift Detection
# - raw/train.csv: dữ liệu mới cần validate
# - training/train.csv: reference data dùng để so sánh drift
# - validated/: thư mục chứa train.csv sau validate (dùng nội bộ làm current_data) và drift_report.html
dvc stage add --force -n calories_validate \
  -d src/pipelines/ml_pipelines/calories_pred/data_val.py \
  -d src/pipelines/ml_pipelines/calories_pred/eval_drift.py \
  -d Artifacts/data/ml_calories/raw/train.csv \
  -d Artifacts/data/ml_calories/training/train.csv \
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
  -p config/pipelines/ml_calories/data_val.yaml: \
  -p config/pipelines/ml_calories/evidently_profile.yaml: \
  -p config/pipelines/global.yaml: \
  -o Artifacts/data/ml_calories/validated \
  -M Artifacts/reports/ml_calories/validation/validation_report.json \
  python src/pipelines/ml_pipelines/calories_pred/data_val.py

# 3. Stage: Data Transformation
dvc stage add --force -n calories_transform \
  -d src/pipelines/ml_pipelines/calories_pred/data_trans.py \
  -d src/pipelines/ml_pipelines/calories_pred/preprocessing.py \
  -d Artifacts/data/ml_calories/validated/train.csv \
  -d Artifacts/data/ml_calories/validated/val.csv \
  -d Artifacts/data/ml_calories/validated/test.csv \
  -d Artifacts/reports/ml_calories/validation/validation_report.json \
  -d Artifacts/reports/ml_calories/evidently/drift_report.json \
  -d src/data_io/dvc_handler.py \
  -d src/nutrition_core/logging/logger.py \
  -d src/nutrition_core/logging/mlflow_tracker.py \
  -d src/nutrition_core/exception/exception.py \
  -p config/pipelines/ml_calories/data_transformation.yaml: \
  -o Artifacts/data/ml_calories/transformed \
  -M Artifacts/reports/ml_calories/transformation/transformation_report.json \
  python src/pipelines/ml_pipelines/calories_pred/data_trans.py

# 4. Stage: Model Training
dvc stage add --force -n calories_train \
  -d src/pipelines/ml_pipelines/calories_pred/model_train.py \
  -d Artifacts/data/ml_calories/transformed/preprocessing.pkl \
  -d Artifacts/data/ml_calories/transformed/train.npy \
  -d Artifacts/data/ml_calories/transformed/val.npy \
  -d Artifacts/reports/ml_calories/transformation/transformation_report.json \
  -d src/data_io/dvc_handler.py \
  -d src/nutrition_core/logging/logger.py \
  -d src/nutrition_core/logging/mlflow_tracker.py \
  -d src/nutrition_core/exception/exception.py \
  -d src/utils/ml_utils/metric/regression_metric.py \
  -d src/utils/ml_utils/model/estimator.py \
  -d src/utils/ml_utils/model/model_factory.py \
  -d src/utils/ml_utils/model/model_trainer.py \
  -p config/pipelines/ml_calories/hydra_config.yaml: \
  -p config/pipelines/ml_calories/train_params.yaml: \
  -o Artifacts/data/ml_calories/training \
  -M Artifacts/reports/ml_calories/training/training_report.json \
  python src/pipelines/ml_pipelines/calories_pred/model_train.py