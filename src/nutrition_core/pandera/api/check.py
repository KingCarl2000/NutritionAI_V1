"""
Module chứa các Custom Checks (quy tắc kiểm tra tùy chỉnh) cho thư viện Pandera,
được thiết kế đặc thù cho các dữ liệu Dinh dưỡng (Nutrition) và Machine Learning.
"""

import pandas as pd
import pandera.pandas as pa
import pandera.extensions as pa_ext

# ==========================================
# 1. ĐĂNG KÝ CUSTOM CHECK TOÀN CỤC (EXTENSIONS API)
# ==========================================
# Những check này có thể gọi trực tiếp dưới dạng tham số trong pa.Field hoặc pa.Check

@pa_ext.register_check_method(statistics=["max_value"])
def is_realistic_calorie_count(pandas_obj: pd.Series, *, max_value: float = 10000.0) -> pd.Series:
    """Kiểm tra lượng calo có nằm trong ngưỡng thực tế không."""
    return (pandas_obj >= 0) & (pandas_obj <= max_value)

@pa_ext.register_check_method(statistics=["min_age", "max_age"])
def is_valid_age(pandas_obj: pd.Series, *, min_age: int = 0, max_age: int = 120) -> pd.Series:
    """Kiểm tra độ tuổi hợp lệ của người dùng."""
    return (pandas_obj >= min_age) & (pandas_obj <= max_age)


# ==========================================
# 2. OBJECT-BASED API: CÁC CHECK ĐỊNH NGHĨA SẴN
# ==========================================
# Sử dụng khi bạn khởi tạo schema theo dạng pa.DataFrameSchema(...) hoặc cấu hình động.

class NutritionChecks:
    """
    Lớp tiện ích chứa các đối tượng pa.Check dùng cho Object-based schema (DataFrameSchema).
    """
    
    @staticmethod
    def check_positive_macronutrients() -> pa.Check:
        """Vectorized Check: Kiểm tra các chất dinh dưỡng (Protein, Fat, Carbs) không âm."""
        return pa.Check(
            lambda s: s >= 0,
            error="Lượng chất dinh dưỡng (macronutrients) không được có giá trị âm."
        )

    @staticmethod
    def check_valid_meal_type() -> pa.Check:
        """Element-wise Check: Kiểm tra loại bữa ăn hợp lệ."""
        valid_meals = {"Breakfast", "Lunch", "Dinner", "Snack"}
        return pa.Check(
            lambda val: val in valid_meals,
            element_wise=True,
            error=f"Loại bữa ăn không hợp lệ. Chỉ chấp nhận: {valid_meals}",
            ignore_na=True  # Bỏ qua kiểm tra nếu giá trị Null (để thuộc tính nullable=True xử lý)
        )

    @staticmethod
    def check_warning_high_sugar() -> pa.Check:
        """Check sinh ra cảnh báo (SchemaWarning) thay vì dừng pipeline báo lỗi nếu đường quá cao."""
        return pa.Check(
            lambda s: s <= 100.0,
            raise_warning=True,
            error="Cảnh báo: Lượng đường (Sugar) cao bất thường (> 100g), cần xem xét lại dữ liệu."
        )


# ==========================================
# 3. CLASS-BASED API: MIXINS CHO QUY TẮC ĐA CỘT
# ==========================================

class NutritionCrossChecks:
    """
    Lớp Mixin chứa các hàm kiểm tra tương quan đa cột (@pa.dataframe_check).
    Các Schema Class (kế thừa DataFrameModel) có thể kế thừa lớp này để dùng chung logic.
    """
    
    @pa.dataframe_check(
        name="macronutrient_calories_consistency", 
        error="Tổng calo từ Protein, Fat, Carbs không khớp với cột tổng Calories."
    )
    def check_macro_calories_consistency(cls, df: pd.DataFrame) -> pd.Series:
        """
        Wide Check: Đảm bảo tổng calo tính từ P-F-C không lệch quá 20% so với cột tổng calories.
        - 1g Protein = 4 kcal
        - 1g Carbs = 4 kcal
        - 1g Fat = 9 kcal
        """
        required_cols = {"protein_g", "carbs_g", "fat_g", "calories"}
        
        # Bỏ qua check nếu DataFrame không chứa đủ các cột này
        if not required_cols.issubset(df.columns):
            return pd.Series([True] * len(df), index=df.index)

        # Tính toán lượng calo lý thuyết
        calculated_calories = (df["protein_g"] * 4) + (df["carbs_g"] * 4) + (df["fat_g"] * 9)
        
        # Cho phép sai số 20% do nhãn dán dinh dưỡng thực tế thường làm tròn
        lower_bound = calculated_calories * 0.8
        upper_bound = calculated_calories * 1.2
        
        # Xử lý trường hợp calo = 0 để tránh lỗi logic
        is_zero_cal = df["calories"] == 0
        is_within_bounds = (df["calories"] >= lower_bound) & (df["calories"] <= upper_bound)
        
        return is_zero_cal | is_within_bounds