import pandera.pandas as pa
from pandera.typing import Series

class MlCaloriesInputSchema(pa.DataFrameModel):
    class Config:
        """Khung Schema sinh tự động từ EDA. CẦN CẬP NHẬT NGƯỠNG SINH HỌC."""
        coerce = True
        strict = True

    User_ID: Series[int] = pa.Field(nullable=False)  # TODO: Xác nhận lại kiểu (int/float) và cập nhật ngưỡng
    Age: Series[int] = pa.Field(nullable=False)  # TODO: Xác nhận lại kiểu (int/float) và cập nhật ngưỡng
    Height: Series[float] = pa.Field(nullable=False)  # TODO: Xác nhận lại kiểu (int/float) và cập nhật ngưỡng
    Weight: Series[float] = pa.Field(nullable=False)  # TODO: Xác nhận lại kiểu (int/float) và cập nhật ngưỡng
    Duration: Series[float] = pa.Field(nullable=False)  # TODO: Xác nhận lại kiểu (int/float) và cập nhật ngưỡng
    Heart_Rate: Series[float] = pa.Field(nullable=False)  # TODO: Xác nhận lại kiểu (int/float) và cập nhật ngưỡng
    Body_Temp: Series[float] = pa.Field(nullable=False)  # TODO: Xác nhận lại kiểu (int/float) và cập nhật ngưỡng
    Gender: Series[str] = pa.Field(nullable=False)  # TODO: Cập nhật danh sách isin=[...]

class MlCaloriesTrainingSchema(MlCaloriesInputSchema):
    Calories: Series[float] = pa.Field(nullable=False)  # TODO: Cập nhật ngưỡng cho Target