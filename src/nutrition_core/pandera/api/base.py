"""
Module base định nghĩa các lớp cơ sở (Base Schemas) và tiện ích kiểm định dữ liệu (Data Validation)
sử dụng Pandera API cho dự án NutritionAI_V1.
"""

from typing import Any, Dict, Optional, Type, TypeVar, Union
import logging
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

logger = logging.getLogger(__name__)

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

        Args:
            df (pd.DataFrame): DataFrame đầu vào cần kiểm định.
            lazy (bool): Nếu True, sẽ gom tất cả các lỗi vi phạm rồi mới báo ngoại lệ
                        thay vì dừng ngay ở lỗi đầu tiên.

        Returns:
            pd.DataFrame: DataFrame đã được kiểm định và ép kiểu (coerce).

        Raises:
            SchemaError / SchemaErrors: Khi dữ liệu không vượt qua các quy tắc kiểm định.
        """
        try:
            logger.info(f"[Pandera] Bắt đầu kiểm định DataFrame với Schema '{cls.__name__}'...")
            validated_df = cls.validate(df, lazy=lazy)
            logger.info(f"[Pandera] Kiểm định dữ liệu thành công cho Schema '{cls.__name__}'.")
            return validated_df
        except (SchemaError, SchemaErrors) as err:
            logger.error(f"[Pandera] Kiểm định dữ liệu thất bại tại '{cls.__name__}': {err}")
            raise err


def build_dataframe_schema(
    columns: Dict[str, pa.Column],
    coerce: bool = True,
    strict: bool = False,
    name: Optional[str] = None,
    **kwargs: Any,
) -> pa.DataFrameSchema:
    """
    Hàm khởi tạo linh hoạt cho Object-based Schema (DataFrameSchema)
    khi cần tạo quy tắc kiểm định theo phương thức imperative.

    Args:
        columns (Dict[str, pa.Column]): Dictionary cấu hình các cột và quy tắc kiểm tra.
        coerce (bool): Tự động ép kiểu dữ liệu.
        strict (bool): Bắt buộc chỉ cho phép các cột đã khai báo.
        name (Optional[str]): Tên nhận diện của Schema.

    Returns:
        pa.DataFrameSchema: Đối tượng schema đã được khởi tạo.
    """
    return pa.DataFrameSchema(
        columns=columns,
        coerce=coerce,
        strict=strict,
        name=name,
        **kwargs
    )


# Custom Base Checks thường dùng cho dữ liệu Dinh dưỡng & ML
class CommonChecks:
    """Các hàm kiểm tra bổ trợ dùng chung cho các đường ống kiểm định dữ liệu."""

    @staticmethod
    def is_non_negative(series: pd.Series) -> pd.Series:
        """Kiểm tra giá trị số không âm (>= 0), phù hợp cho calories, weight, height, v.v."""
        return series >= 0

    @staticmethod
    def is_valid_percentage(series: pd.Series) -> pd.Series:
        """Kiểm tra tỷ lệ phần trăm (từ 0.0 đến 100.0 hoặc 0.0 đến 1.0)."""
        return (series >= 0.0) & (series <= 100.0)