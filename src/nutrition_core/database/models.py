from sqlalchemy import Column, String, Float, Integer, Index
from src.nutrition_core.database.session import Base

# Model cho Dataset Calories
class CalorieLog(Base):
    __tablename__ = "calories"
    __table_args__ = {'schema': 'nutrition_ai'}

    # Đánh B-Tree index đơn giản cho User_ID
    User_ID = Column(String, index=True) 
    ActivityDay = Column(String)
    Calories = Column(Float)


    User_ID = Column(String)
    ActivityHour = Column(String)
    StepTotal = Column(Integer)