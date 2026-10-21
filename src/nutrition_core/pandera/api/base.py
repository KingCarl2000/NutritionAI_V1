"""
Module base định nghĩa các lớp cơ sở (Base Schemas) và tiện ích kiểm định dữ liệu (Data Validation)
sử dụng Pandera API cho dự án NutritionAI_V1.
"""
from typing import Any, Dict, Optional, Type, TypeVar, Union
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

# Import custom logger và MLOpsTracker
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker

# Khởi tạo MLOpsTracker (bạn có thể đưa experiment_name ra file config nếu cần)
mlflow_tracker = MLOpsTracker(experiment_name="Data_Validation_Pipeline")

# Type hint chung cho các Class kế thừa từ pa.DataFrameModel
SchemaModelT = TypeVar("SchemaModelT", bound=pa.DataFrameModel)

class BaseSchema(pa.DataFrameModel):
    """
    Lớp Schema cơ sở cho tất cả các mô hình dữ liệu trong NutritionAI_V1.
    Mọi Schema dữ liệu (Calories, Customer Clustering, Features,...) nên kế thừa lớp này.
    """

    class Config:
        coerce = True      # Tự động ép kiểu về đúng dtype khai báo
        strict = False     # Mặc định linh hoạt (cho phép cột phụ). Có thể ghi đè sang True ở lớp con
        ordered = False    # Không bắt buộc thứ tự cột trong DataFrame

    @classmethod
    def validate_df(
        cls: Type[SchemaModelT],
        df: pd.DataFrame,
        lazy: bool = True
    ) -> pd.DataFrame:
        """
        Thực thi kiểm định một Pandas DataFrame theo Schema.
        """
        schema_name = cls.__name__
        try:
            logger.info(f"[Pandera] Bắt đầu kiểm định DataFrame với Schema '{schema_name}'...")
            validated_df = cls.validate(df, lazy=lazy)
            logger.info(f"[Pandera] Kiểm định dữ liệu thành công cho Schema '{schema_name}'.")
            
            # Ghi log metrics vào MLflow thông qua MLOpsTracker
            mlflow_tracker.log_params_and_metrics(
                params={"validation_schema": schema_name, "validation_status": "success"},
                metrics={f"validated_{schema_name}_rows": len(validated_df)}
            )
            
            return validated_df

        except (SchemaError, SchemaErrors) as err:
            error_msg = f"[Pandera] Kiểm định dữ liệu thất bại tại '{schema_name}': {err}"
            
            # Ghi log lỗi vào file (logger.py) và push error tag lên MLflow
            mlflow_tracker.log_error_and_tag(
                error_msg=error_msg,
                tags={"validation_schema": schema_name, "stage": "pandera_validation"}
            )
            
            raise err

def build_dataframe_schema(
    columns: Dict[str, pa.Column],
    coerce: bool = True,
    strict: bool = False,
    name: Optional[str] = None,
    **kwargs: Any,
) -> pa.DataFrameSchema:
    """Hàm khởi tạo linh hoạt cho Object-based Schema (DataFrameSchema)."""
    return pa.DataFrameSchema(
        columns=columns,
        coerce=coerce,
        strict=strict,
        name=name,
        **kwargs
    )

class CommonChecks:
    """Các hàm kiểm tra bổ trợ dùng chung cho các đường ống kiểm định dữ liệu."""

    @staticmethod
    def is_non_negative(series: pd.Series) -> pd.Series:
        """Kiểm tra giá trị số không âm (>= 0)."""
        return series >= 0

    @staticmethod
    def is_valid_percentage(series: pd.Series) -> pd.Series:
        """Kiểm tra tỷ lệ phần trăm (từ 0.0 đến 100.0)."""
        return (series >= 0.0) & (series <= 100.0)