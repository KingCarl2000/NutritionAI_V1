import sys
from src.nutrition_core.logging import logger
from types import ModuleType

class NutritionBaseException(Exception):
    """Class base chứa logic lấy thông tin lỗi chung cho toàn hệ thống."""
    def __init__(self, error_message, error_details: ModuleType):
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()
        
        if exc_tb is not None:
            self.lineno = exc_tb.tb_lineno
            self.file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            self.lineno = None
            self.file_name = None

    def __str__(self):
        return "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name, self.lineno, str(self.error_message)
        )

class DataPipelineException(NutritionBaseException):
    """Lỗi chung cho quá trình luân chuyển và nạp dữ liệu."""
    pass

class DataTransformationError(DataPipelineException):
    """
    Lỗi phát sinh trong quá trình dbt/SQL biến đổi dữ liệu.
    Ví dụ: Lọc nhầm các đặc trưng (features) có độ tương quan thấp nhưng 
    lại cần thiết cho các phép biến đổi hồi quy đa thức (polynomial regression).
    """
    pass

class DataCleansingError(DataPipelineException):
    """
    Lỗi phát sinh khi làm sạch dữ liệu.
    Ví dụ: Vòng lặp thay thế giá trị null tự động ghi đè sai mã định danh 
    của các cột phân loại (categorical columns).
    """
    pass