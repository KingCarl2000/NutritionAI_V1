import os
import pandas as pd
import pandera as pa
from pandera.errors import SchemaErrors

# Import schema chuẩn từ module config (loại bỏ việc tự định nghĩa lại)
from config.pipelines.ml_calories.calories_schema import (
    MlCaloriesInputSchema,
    MlCaloriesTrainingSchema
)

# Tận dụng các module Core đã được định nghĩa trong dự án
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker

# =============================================================================
# MLOPS PIPELINE: VALIDATION, QUARANTINE ROUTING & ALERTING
# =============================================================================

def run_validation_pipeline(
    df: pd.DataFrame, 
    mode: str = "train", 
    tracker: MLOpsTracker = None,
    error_threshold: float = 0.05  # Thêm ngưỡng cảnh báo mặc định 5%
) -> pd.DataFrame:
    """
    Thực thi Data Validation, phân tách dữ liệu lỗi (Quarantine) và kích hoạt Alert.

    Args:
        df: DataFrame cần kiểm tra.
        mode: 'train' (sử dụng MlCaloriesTrainingSchema) hoặc 'inference' (sử dụng MlCaloriesInputSchema).
        tracker: Đối tượng MLOpsTracker để ghi nhận metrics lên MLflow.
        error_threshold: Ngưỡng tỷ lệ lỗi cho phép trước khi kích hoạt Alert chặn pipeline.

    Returns:
        pd.DataFrame: Tập dữ liệu Valid Data đã được làm sạch.
    """
    logger.info(f"Bắt đầu quá trình Data Validation (Chế độ: {mode.upper()})...")

    # Lựa chọn Schema cấu hình sẵn dựa trên mode
    schema = MlCaloriesTrainingSchema if mode == "train" else MlCaloriesInputSchema

    try:
        # A. Kích hoạt Lazy Validation để bắt toàn bộ lỗi
        valid_df = schema.validate(df, lazy=True)
        logger.info(f"Validation thành công 100%. {len(valid_df)} bản ghi hợp lệ.")

        if tracker:
            tracker.log_params_and_metrics(metrics={f"val_{mode}_error_rate": 0.0})

        return valid_df

    except SchemaErrors as err:
        # B. Quy trình Cô lập Dữ liệu (Quarantine Routing)
        failure_indices = err.failure_cases['index'].dropna().unique()

        # Phân tách 2 luồng: Valid và Invalid
        invalid_df = df.loc[failure_indices]
        valid_df = df.drop(index=failure_indices)

        total_records = len(df)
        invalid_count = len(invalid_df)
        error_rate = invalid_count / total_records

        logger.warning(
            f"Phát hiện {invalid_count}/{total_records} bản ghi vi phạm logic "
            f"(Tỷ lệ lỗi: {error_rate:.2%})."
        )

        if tracker:
            tracker.log_params_and_metrics(
                metrics={
                    f"val_{mode}_error_rate": error_rate,
                    f"val_{mode}_invalid_count": invalid_count
                }
            )

        # C. Định tuyến lưu trữ vào vùng cách ly
        quarantine_dir = os.path.join("Artifacts", "reports", "ml_calories", "validation")
        os.makedirs(quarantine_dir, exist_ok=True)

        quarantine_data_path = os.path.join(quarantine_dir, f"quarantine_{mode}_data.csv")
        invalid_df.to_csv(quarantine_data_path, index=False)

        report_path = os.path.join(quarantine_dir, f"error_report_{mode}.csv")
        err.failure_cases.to_csv(report_path, index=False)
        
        logger.info(f"Đã lưu dữ liệu lỗi tại {quarantine_data_path} và báo cáo tại {report_path}")

        # D. Xử lý Alert: Chặn pipeline nếu tỷ lệ lỗi vượt ngưỡng
        if error_rate > error_threshold:
            alert_msg = f"DATA ALERT: Tỷ lệ lỗi ({error_rate:.2%}) vượt ngưỡng cho phép ({error_threshold:.2%}). Dừng pipeline!"
            logger.error(alert_msg)
            raise ValueError(alert_msg) # Bắn ngoại lệ để dừng tiến trình ETL

        return valid_df