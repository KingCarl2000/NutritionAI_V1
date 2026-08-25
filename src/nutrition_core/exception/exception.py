import sys
from types import ModuleType
from src.nutrition_core.logging import logger

class NutritionBaseException(Exception):
    """Class base chứa lỗi hệ thống cho NutritionAI_V1."""
    
    def __init__(self, error_message: str, error_details: ModuleType = sys):
        super().__init__(error_message)
        self.error_message = error_message
        self.file_name, self.lineno = self._get_detailed_error_info(error_details)

    @staticmethod
    def _get_detailed_error_info(error_details: ModuleType):
        """Trích xuất tên file và số dòng xảy ra lỗi từ traceback."""
        try:
            _, _, exc_tb = error_details.exc_info()
            if exc_tb is not None:
                file_name = exc_tb.tb_frame.f_code.co_filename
                lineno = exc_tb.tb_lineno
                return file_name, lineno
        except Exception:
            pass
        return "Unknown", "Unknown"

    def __str__(self):
        return "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name, self.lineno, str(self.error_message)
        )

# ==========================================
# DATA PIPELINE EXCEPTIONS
# ==========================================

class DataPipelineException(NutritionBaseException):
    """Lỗi chung cho quá trình luân chuyển và nạp dữ liệu."""
    pass

class DataTransformationError(DataPipelineException):
    """Lỗi phát sinh trong quá trình biến đổi dữ liệu (ví dụ: vô tình loại bỏ các đặc trưng cần thiết cho polynomial regression)."""
    pass

class DataCleansingError(DataPipelineException):
    """Lỗi phát sinh khi làm sạch dữ liệu (ví dụ: vòng lặp thay thế null ghi đè sai mã categorical columns)."""
    pass


# ==========================================
# ML PIPELINE EXCEPTIONS
# ==========================================

class MLPipelineException(NutritionBaseException):
    """Lỗi chung cho các ML Pipelines (calorie_prediction, clustering_customer, v.v.)."""
    pass

class ModelTrainingError(MLPipelineException):
    """Lỗi phát sinh trong quá trình huấn luyện mô hình (ví dụ: sai tham số khi thiết lập K-means, DBSCAN, hoặc Hierarchical clustering)."""
    pass

class ModelEvaluationError(MLPipelineException):
    """Lỗi khi đánh giá hoặc kiểm thử mô hình trên tập validation/test."""
    pass

"""
# ==========================================
# LLM PIPELINE EXCEPTIONS
# ==========================================

class LLMPipelineException(NutritionBaseException):
    """Lỗi chung cho các luồng xử lý NLP và mô hình ngôn ngữ lớn (LLM)."""
    pass

class LLMInferenceError(LLMPipelineException):
    """Lỗi phát sinh trong quá trình suy luận của LLM (ví dụ: vượt quá giới hạn VRAM khi chạy các mô hình local như Qwen 3.5)."""
    pass

class PromptProcessingError(LLMPipelineException):
    """Lỗi khi xử lý chuỗi đầu vào hoặc kết xuất metadata cho template response."""
    pass

"""
