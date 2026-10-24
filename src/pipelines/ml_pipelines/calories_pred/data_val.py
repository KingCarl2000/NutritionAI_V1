import os
import pandas as pd
import pandera as pa
from pandera.errors import SchemaErrors

# 1. Tận dụng Core Logging & Decorator từ kiến trúc NutritionAI_V1
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.pandera.decorators import nutrition_check_io

# 2. Tận dụng Pandera Core API (đã định nghĩa sẵn)
from src.nutrition_core.pandera.api.base import BaseSchema
from src.nutrition_core.pandera.api.check import GenericCheckFactory
# 3. Tận dụng Exception Handle chuẩn của project
from src.nutrition_core.exception.val_exception import DataValidationError

# Cấu hình đường dẫn lưu Quarantine Data (Tham chiếu theo project_tree.md)
QUARANTINE_DIR = "Artifacts/reports/ml_calories/validation/quarantine"
os.makedirs(QUARANTINE_DIR, exist_ok=True)

# ---------------------------------------------------------
# Định nghĩa Schema (Kế thừa từ API core)
# ---------------------------------------------------------
class CaloriesInferenceSchema(BaseSchema):
    class Config:
        coerce = True
        strict = False

    Age: pa.typing.Series[int] = pa.Field(ge=1, coerce=True)
    Height: pa.typing.Series[float] = pa.Field(ge=0.0, coerce=True)
    Weight: pa.typing.Series[float] = pa.Field(ge=0.0, coerce=True)
    Duration: pa.typing.Series[float] = pa.Field(gt=0.0, coerce=True)
    Heart_Rate: pa.typing.Series[float] = pa.Field(ge=40.0, le=220.0, coerce=True)

    class Config:
        coerce = True
        strict = False  # Cho phép các cột phụ như User_ID, Gender đi qua

    # Tận dụng các Custom Check đa cột đã viết trong src/nutrition_core/pandera/api/check.py
    @pa.dataframe_check
    def validate_physiology(cls, df: pd.DataFrame) -> pa.typing.Series[bool]:
        return GenericCheckFactory.check_physiological_limits(df)

    @pa.dataframe_check
    def validate_macronutrients(cls, df: pd.DataFrame) -> pa.typing.Series[bool]:
        # Tự động gọi check tương quan vĩ mô nếu dataset có chứa cột liên quan
        return check_macronutrients_correlation(df)


class CaloriesTrainingSchema(CaloriesInferenceSchema):
    """
    Schema kiểm tra dữ liệu huấn luyện. 
    Kế thừa CaloriesInferenceSchema và bổ sung nhãn mục tiêu.
    """
    Calories: pa.typing.Series[float] = pa.Field(gt=0.0, nullable=False, coerce=True)


# ---------------------------------------------------------
# Xử lý Lỗi và Cô lập Dữ liệu Hỏng (Quarantine Data)
# ---------------------------------------------------------
def _quarantine_invalid_data(e: SchemaErrors, df: pd.DataFrame, context: str) -> pd.DataFrame:
    """
    Trích xuất failure_cases, ghi log, xuất tệp Quarantine và loại bỏ dòng lỗi.
    """
    failures = e.failure_cases
    logger.error(f"[{context}] Phát hiện {len(failures)} lỗi vi phạm Schema. Trích xuất failure_cases...")

    # Tìm index dòng vi phạm
    invalid_indices = failures['index'].dropna().unique()
    invalid_df = df.loc[invalid_indices]

    # Ghi file CSV Quarantine
    quarantine_file = os.path.join(QUARANTINE_DIR, f"{context}_invalid_rows.csv")
    invalid_df.to_csv(quarantine_file, index=False)
    logger.warning(f"[{context}] Đã cách ly {len(invalid_df)} dòng hỏng tại: {quarantine_file}")

    # Làm sạch: Drop invalid rows
    clean_df = df.drop(index=invalid_indices)
    logger.info(f"[{context}] Đã tự động loại bỏ lỗi. Kích thước tập dữ liệu sạch: {clean_df.shape}")

    return clean_df


# ---------------------------------------------------------
# Entry Points cho Pipeline
# ---------------------------------------------------------

# Tận dụng decorator @nutrition_check_io để validate input/output logging
@nutrition_check_io(head=5, tail=5)
def validate_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kiểm tra tập dữ liệu huấn luyện (yêu cầu cột Calories).
    """
    logger.info("[Validation] Bắt đầu quét tập dữ liệu Training với lazy=True...")
    try:
        clean_df = CaloriesTrainingSchema.validate(df, lazy=True)
        logger.info("[Validation] Training Data hợp lệ 100%.")
        return clean_df
    except SchemaErrors as e:
        return _quarantine_invalid_data(e, df, context="training_calories")
    except Exception as e:
        raise DataValidationError(f"Lỗi hệ thống khi validate Training Data: {str(e)}") from e


@nutrition_check_io(head=5, tail=5)
def validate_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kiểm tra tập dữ liệu dự đoán thực tế (không có cột Calories).
    """
    logger.info("[Validation] Bắt đầu quét tập dữ liệu Inference với lazy=True...")
    try:
        clean_df = CaloriesInferenceSchema.validate(df, lazy=True)
        logger.info("[Validation] Inference Data hợp lệ 100%.")
        return clean_df
    except SchemaErrors as e:
        return _quarantine_invalid_data(e, df, context="inference_calories")
    except Exception as e:
        raise DataValidationError(f"Lỗi hệ thống khi validate Inference Data: {str(e)}") from e