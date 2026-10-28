import json
import os
import sys
import yaml
import pandas as pd
from evidently import Report
from evidently.core.datasets import Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset

# Tích hợp Logger tùy chỉnh của NutritionAI_V1 
# Sẽ tự động phân luồng vào oper_logs, warn_logs, hoặc error_logs
from src.nutrition_core.logging.logger import logger

def load_global_thresholds(config_path: str = "config/pipelines/global.yaml") -> dict:
    """Đọc các ngưỡng cấu hình Drift động từ file YAML của dự án."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("mlops_thresholds", {}).get("data_drift", {})

def execute_data_drift_detection(
    reference_path: str = "Artifacts/data/ml_calories/processed/train.csv",
    current_path: str = "Artifacts/data/ml_calories/processed/current_data.csv",
    output_dir: str = "Artifacts/reports/ml_calories/evidently",
    config_path: str = "config/pipelines/global.yaml"
) -> dict:
    """Thực thi kiểm tra Data Drift và xuất báo cáo MLOps."""
    
    # 1. Đọc ngưỡng cấu hình linh hoạt từ global.yaml
    drift_cfg = load_global_thresholds(config_path)
    p_value_alpha = drift_cfg.get("p_value_alpha", 0.05)
    drift_share_threshold = drift_cfg.get("psi_critical", 0.25)

    logger.info("--- BẮT ĐẦU ĐÁNH GIÁ DATA DRIFT ---")

    # 2. Đọc và nạp dữ liệu Tham chiếu / Hiện tại
    if not os.path.exists(reference_path) or not os.path.exists(current_path):
        err_msg = f"Không tìm thấy tệp dữ liệu đầu vào: '{reference_path}' hoặc '{current_path}'"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    logger.info(f"📥 Đang tải tập dữ liệu tham chiếu (Reference): {reference_path}")
    reference_df = pd.read_csv(reference_path)
    logger.info(f"📥 Đang tải tập dữ liệu hiện tại (Current): {current_path}")
    current_df = pd.read_csv(current_path)

    # Đảm bảo thư mục Artifacts của DVC tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # 3. Định nghĩa Lược đồ Dữ liệu (Có thể đồng bộ tiếp với schema.yaml sau này)
    num_cols = reference_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = reference_df.select_dtypes(include=["object", "category"]).columns.tolist()

    data_def = DataDefinition(
        numerical_columns=num_cols, 
        categorical_columns=cat_cols
    )
    ref_dataset = Dataset.from_pandas(reference_df, data_definition=data_def)
    curr_dataset = Dataset.from_pandas(current_df, data_definition=data_def)

    # 4. Khởi tạo & Kích hoạt Báo cáo DataDriftPreset
    logger.info("⚙️ Đang kích hoạt mô-đun Evidently Data Drift Engine...")
    drift_report = Report(metrics=[
        DataSummaryPreset(),
        DataDriftPreset(
            drift_share=drift_share_threshold, 
            threshold=p_value_alpha,           
            num_method="wasserstein",          
            cat_method="chisquare"             
        )
    ])

    snapshot = drift_report.run(
        current_data=curr_dataset, 
        reference_data=ref_dataset
    )

    # 5. Xuất Báo cáo & Lưu Artifacts chuẩn cấu trúc DVC
    html_out = os.path.join(output_dir, "drift_report.html")
    json_out = os.path.join(output_dir, "drift_metrics.json")

    snapshot.save_html(html_out)
    json_data = snapshot.json()
    
    with open(json_out, "w", encoding="utf-8") as f:
        f.write(json_data)

    logger.info(f"✓ Báo cáo trực quan HTML đã lưu tại: {html_out}")
    logger.info(f"✓ Dữ liệu báo cáo JSON đã lưu tại: {json_out}")