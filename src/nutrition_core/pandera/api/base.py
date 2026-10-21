"""
Module base định nghĩa các lớp cơ sở (Base Schemas) và tiện ích kiểm định dữ liệu dùng chung (Generic Validation Rules)
sử dụng Pandera API cho hệ thống NutritionAI_V1.
"""
from typing import Any, Dict, Optional, Type, TypeVar, Union
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

# Import custom Logger và MLOpsTracker từ thư mục logging của dự án
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker

# Khởi tạo MLOpsTracker quản lý tiến trình Data Validation
mlflow_tracker = MLOpsTracker(experiment_name="Data_Validation_Pipeline")

# Type hint chung cho các Class kế thừa từ pa.DataFrameModel
SchemaModelT = TypeVar("SchemaModelT", bound=pa.DataFrameModel)


class BaseSchema(pa.DataFrameModel):
    """
    Lớp Schema cơ sở áp dụng các quy tắc kiểm định khung cho tất cả mô hình dữ liệu trong NutritionAI_V1.
    Mọi Schema lớp con (như CaloriesSchema, UserSchema,...) đều nên kế thừa lớp này.
    """

    class Config:
        coerce = True                 # [Quy tắc 2] Tự động ép kiểu dữ liệu về đúng dtype khai báo
        strict = False                # [Quy tắc 1] Linh hoạt chấp nhận các cột phụ ở tầng Base (filter ở pipeline chi tiết)
        ordered = False               # [Quy tắc 1] Không bắt buộc thứ tự cột trừ khi pipeline cụ thể yêu cầu
        unique_column_names = True    # [Quy tắc 1] Đảm bảo tên các cột trong DataFrame không bị trùng lặp
        add_missing_columns = True    # [Quy tắc 3] Tự động tạo cột và điền default value nếu dữ liệu thiếu cột không bắt buộc

    @classmethod
    def validate_df(
        cls: Type[SchemaModelT],
        df: pd.DataFrame,
        lazy: bool = True
    ) -> pd.DataFrame:
        """
        Thực thi kiểm định một Pandas DataFrame theo Base Schema & Chiến lược gom lỗi (Lazy Validation).

        Args:
            df (pd.DataFrame): DataFrame đầu vào cần kiểm định.
            lazy (bool): Thu thập tất cả vi phạm trước khi báo lỗi (mặc định True).

        Returns:
            pd.DataFrame: DataFrame đã được kiểm định và ép kiểu hợp lệ.

        Raises:
            SchemaErrors / SchemaError: Khi dữ liệu vi phạm các quy tắc kiểm định.
        """
        schema_name = cls.__name__
        total_rows = len(df)
        logger.info(f"[Pandera] Bắt đầu kiểm định DataFrame ({total_rows} dòng) với Schema '{schema_name}'...")

        try:
            # Thực thi kiểm định dữ liệu
            validated_df = cls.validate(df, lazy=lazy)
            logger.info(f"[Pandera] Kiểm định dữ liệu THÀNH CÔNG cho Schema '{schema_name}'.")

            # [Quy tắc 7] Ghi nhận chỉ số thành công lên MLflow
            mlflow_tracker.log_params_and_metrics(
                params={
                    f"validation_{schema_name}_status": "SUCCESS",
                    f"validation_{schema_name}_schema": schema_name
                },
                metrics={
                    f"validation_{schema_name}_rows": len(validated_df)
                }
            )

            return validated_df

        except SchemaErrors as err:
            # [Quy tắc 5] Bóc tách và trích xuất danh sách các dòng/cột vi phạm từ failure_cases
            failure_cases_df = err.failure_cases
            total_errors = len(failure_cases_df)
            error_msg = f"[Pandera] Kiểm định THẤT BẠI tại '{schema_name}': Phát hiện {total_errors} trường hợp vi phạm quy tắc."

            # [Quy tắc 7] Ghi vết lỗi ra file log và console
            logger.error(error_msg)
            logger.error(f"[Pandera] Trích xuất 10 vi phạm đầu tiên:\n{failure_cases_df.head(10)}")

            # [Quy tắc 7] Đẩy thông số lỗi và tag FAILED lên MLflow
            mlflow_tracker.log_error_and_tag(
                error_msg=error_msg,
                tags={
                    "validation_schema": schema_name,
                    "validation_status": "FAILED",
                    "total_failure_cases": str(total_errors)
                }
            )
            raise err

        except SchemaError as err:
            error_msg = f"[Pandera] Lỗi Schema đơn lẻ tại '{schema_name}': {err}"
            logger.error(error_msg)

            mlflow_tracker.log_error_and_tag(
                error_msg=error_msg,
                tags={
                    "validation_schema": schema_name,
                    "validation_status": "FAILED"
                }
            )
            raise err


def build_dataframe_schema(
    columns: Dict[str, pa.Column],
    coerce: bool = True,
    strict: Union[bool, str] = False,
    unique_column_names: bool = True,
    add_missing_columns: bool = True,
    name: Optional[str] = None,
    **kwargs: Any,
) -> pa.DataFrameSchema:
    """
    Hàm khởi tạo linh hoạt cho Object-based Schema (DataFrameSchema) áp dụng các cấu hình khung chuẩn.

    Args:
        columns (Dict[str, pa.Column]): Cấu hình quy tắc kiểm tra từng cột.
        coerce (bool): Ép kiểu dữ liệu tự động.
        strict (Union[bool, str]): Chính sách xử lý cột ngoài danh mục.
        unique_column_names (bool): Đảm bảo tính duy nhất của tên cột.
        add_missing_columns (bool): Tự động thêm cột bị thiếu nếu có default.
        name (Optional[str]): Tên định danh cho Schema.

    Returns:
        pa.DataFrameSchema: Đối tượng schema imperatively khởi tạo.
    """
    return pa.DataFrameSchema(
        columns=columns,
        coerce=coerce,
        strict=strict,
        unique_column_names=unique_column_names,
        add_missing_columns=add_missing_columns,
        name=name,
        **kwargs
    )


class CommonChecks:
    """
    [Quy tắc 4] Tập hợp các hàm kiểm định tĩnh cơ bản (Generic Validation Logic)
    dùng chung cho mọi tập dữ liệu trong hệ thống.
    """

    @staticmethod
    def is_non_negative(series: pd.Series) -> pd.Series:
        """Kiểm tra giá trị số không âm (>= 0), áp dụng cho chỉ số đo lường thực tế."""
        return series >= 0

    @staticmethod
    def is_valid_percentage(series: pd.Series) -> pd.Series:
        """Kiểm tra tỷ lệ phần trăm nằm trong khoảng hợp lệ từ 0.0 đến 100.0."""
        return (series >= 0.0) & (series <= 100.0)

    @staticmethod
    def is_unique_identifier(series: pd.Series) -> bool:
        """Kiểm tra tính duy nhất của cột định danh khóa chính (Primary Key / ID)."""
        return series.is_unique