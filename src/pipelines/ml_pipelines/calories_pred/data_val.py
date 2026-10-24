import os
import pandas as pd
import pandera as pa
from pandera.typing import Series
from pandera.errors import SchemaErrors

# Tận dụng các module Core đã được định nghĩa trong dự án
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker

# =============================================================================
# 1 & 2. ĐỊNH NGHĨA SCHEMA VÀ RÀNG BUỘC (PHYSIOLOGICAL BOUNDS)
# =============================================================================

class InferenceSchema(pa.DataFrameModel):
    """
    Nhiệm vụ 2: Schema cho tập Inference (8 trường dữ liệu đầu vào).
    Loại bỏ hoàn toàn cột nhãn `Calories`.
    """
    class Config:
        strict = True
        coerce = True  # Ép kiểu tự động (VD: int -> str cho User_ID)

    # Đặc trưng định danh và phân loại
    User_ID: Series[str]  # Xử lý dưới dạng chuỗi để tránh bị hiểu là số học
    Gender: Series[str] = pa.Field(isin=["Nam", "Nữ", "Male", "Female"])

    # Kiểm tra giới hạn sinh lý cơ bản (Baseline BMI metrics)
    Age: Series[int] = pa.Field(ge=0, le=150)
    Height: Series[float] = pa.Field(ge=120.0, le=220.0)
    Weight: Series[float] = pa.Field(ge=30.0, le=200.0)
    Duration: Series[float] = pa.Field(ge=0.0)

    # Kiểm tra giới hạn y khoa thực tế
    Heart_Rate: Series[float] = pa.Field(ge=50.0, le=220.0)
    Body_Temp: Series[float] = pa.Field(ge=36.0, le=40.0)


class TrainingSchema(InferenceSchema):
    """
    Nhiệm vụ 1: Schema cho tập Training (9 trường dữ liệu).
    Kế thừa toàn bộ đặc trưng từ InferenceSchema và bổ sung nhãn mục tiêu.
    """
    Calories: Series[float] = pa.Field(ge=0.0)


# =============================================================================
# 3. MLOPS PIPELINE: VALIDATION, QUARANTINE ROUTING & ALERTING
# =============================================================================

def run_validation_pipeline(
    df: pd.DataFrame, 
    mode: str = "train", 
    tracker: MLOpsTracker = None
) -> pd.DataFrame:
    """
    Thực thi Data Validation, phân tách dữ liệu lỗi (Quarantine) và kích hoạt Alert.
    
    Args:
        df: DataFrame cần kiểm tra.
        mode: 'train' (sử dụng TrainingSchema) hoặc 'inference' (sử dụng InferenceSchema).
        tracker: Đối tượng MLOpsTracker để ghi nhận metrics lên MLflow.
        
    Returns:
        pd.DataFrame: Tập dữ liệu Valid Data đã được làm sạch.
    """
    logger.info(f"Bắt đầu quá trình Data Validation (Chế độ: {mode.upper()})...")
    
    # Lựa chọn Schema dựa trên mode
    schema = TrainingSchema if mode == "train" else InferenceSchema
    
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

        # Định tuyến lưu trữ vào vùng cách ly
        quarantine_dir = os.path.join("Artifacts", "reports", "ml_calories", "validation")
        os.makedirs(quarantine_dir, exist_ok=True)
        
        # C. Xuất báo cáo tổng hợp các trường hợp vi phạm
        quarantine_data_path = os.path.join(quarantine_dir, f"quarantine_{mode}_data.csv")
        report_path = os.path.join(quarantine_dir, f"val_{mode}_report.csv")
        
        invalid_df.to_csv(quarantine_data_path, index=False)
        err.failure_cases.to_csv(report_path, index=False)
        
        logger.info(f"[Quarantine] Dữ liệu hỏng đã được cách ly tại: {quarantine_data_path}")
        logger.info(f"[Report] Báo cáo chi tiết nguyên nhân lưu tại: {report_path}")

        # C. Cảnh báo (Alerting) với ngưỡng 5%
        if error_rate > 0.05:
            logger.critical(
                f"[CRITICAL ALERT] Tỷ lệ dữ liệu hỏng ({error_rate:.2%}) vượt ngưỡng an toàn 5%. "
                f"Chất lượng dữ liệu đầu vào đang suy giảm nghiêm trọng!"
            )
            # Tùy thuộc vào thiết kế, có thể sử dụng 'raise' tại đây để ngắt cứng Pipeline
            # raise ValueError("Validation Error Rate exceeded Critical Threshold (5%).")
        
        return valid_df