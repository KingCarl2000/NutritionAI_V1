import pandera.pandas as pa
from pandera.typing import Series

class MlCaloriesInputSchema(pa.DataFrameModel):
    class Config:
        """Schema kiểm định dữ liệu đầu vào dựa trên ngưỡng sinh học thực tế."""
        coerce = True
        strict = True

    # User_ID: Chỉ cần là số nguyên dương
    User_ID: Series[int] = pa.Field(ge=1)  
    
    # Gender: Giới hạn trong các phân loại giới tính cơ bản
    Gender: Series[str] = pa.Field(isin=["male", "female", "other"])
    
    # Age: Độ tuổi có thể tập luyện thể thao (từ 0 đến 200 tuổi)
    Age: Series[int] = pa.Field(ge=0, le=200)
    
    # Height (Chiều cao - cm): Ngưỡng từ người lùn nhất đến người cao nhất thế giới (xấp xỉ)
    Height: Series[float] = pa.Field(ge=40, le=255.0)
    
    # Weight (Cân nặng - kg): Ngưỡng từ trẻ em/người suy dinh dưỡng đến người béo phì nặng
    Weight: Series[float] = pa.Field(ge=2, le=300.0)
    
    # Duration (Thời gian tập - phút): Từ 0 phút đến 300 phút (5 tiếng chạy marathon)
    Duration: Series[float] = pa.Field(ge=0, le=300.0)
    
    # Heart_Rate (Nhịp tim - bpm): Từ trạng thái nghỉ ngơi của vđv (40) đến nhịp tim tối đa (220)
    Heart_Rate: Series[float] = pa.Field(ge=40.0, le=220.0)
    
    # Body_Temp (Nhiệt độ cơ thể - °C): Phủ phổ nhiệt độ y khoa từ hạ thân nhiệt đến sốt cao/tập luyện nặng
    Body_Temp: Series[float] = pa.Field(ge=35.0, le=42.5)


class MlCaloriesTrainingSchema(MlCaloriesInputSchema):
    """Schema dành riêng cho Training Pipeline (Kế thừa Input Schema + Cột Target)."""
    
    # Calories (Lượng calo tiêu thụ - kcal): Từ 1 kcal đến 5000 kcal cho một phiên tập luyện cực nặng
    Calories: Series[float] = pa.Field(ge=1.0, le=5000.0)