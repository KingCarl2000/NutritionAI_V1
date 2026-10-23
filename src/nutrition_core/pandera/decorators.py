"""
Module chứa các Decorators hỗ trợ kiểm định dữ liệu cho Data Pipelines trong NutritionAI_V1.
Bọc các decorator gốc của Pandera nhằm chuẩn hóa ngoại lệ (DataValidationError) 
và tự động ghi nhận (tracking) log/metadata vào MLflow.
"""

import functools
from typing import Callable, Any

import pandera as pa
from pandera.errors import SchemaError, SchemaErrors

# Import các module cốt lõi của dự án
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker
from src.nutrition_core.exception.val_exception import DataValidationError, PanderaErrorHandler
from src.nutrition_core.pandera.backends.pandas_executor import NutritionPandasBackend

# Khởi tạo tracker quản lý tiến trình validation trong pipeline
mlflow_tracker = MLOpsTracker(experiment_name="Data_Validation_Pipeline")


def _log_validation_failure(func_name: str, errors: Any) -> None:
    """Call the tracker failure hook when it is available."""
    log_failure = getattr(mlflow_tracker, "log_validation_failure", None)
    if callable(log_failure):
        log_failure(func_name=func_name, errors=errors)

def nutrition_check_types(*args, **kwargs):
    """
    Decorator tự động kiểm tra dữ liệu dựa trên Type Annotations (DataFrame[SchemaModel]).
    Tích hợp bắt lỗi SchemaError/SchemaErrors từ Pandera và chuyển thành DataValidationError.
    
    Args:
        Các tham số tương tự `pandera.check_types` (ví dụ: lazy=True).
    """
    def decorator(func: Callable) -> Callable:
        # Áp dụng decorator gốc của Pandera để lấy logic kiểm tra type hint
        pandera_decorator = pa.check_types(*args, **kwargs)
        wrapped_func = pandera_decorator(func)

        @functools.wraps(func)
        def wrapper(*func_args, **func_kwargs):
            func_name = func.__name__
            logger.info(f"[Validation Start] Bắt đầu kiểm định hàm pipeline: {func_name}")
            
            try:
                # Thực thi hàm đã được bọc logic kiểm định
                result = wrapped_func(*func_args, **func_kwargs)
                logger.info(f"[Validation Success] Dữ liệu hợp lệ qua hàm: {func_name}")
                return result
                
            except (SchemaError, SchemaErrors) as e:
                logger.error(f"[Validation Failed] Phát hiện dữ liệu không hợp lệ tại: {func_name}")
                
                # Chuyển đổi chi tiết lỗi từ Pandera sang format chuẩn của dự án
                error_details = getattr(PanderaErrorHandler, "extract_errors")(e)
                
                # Ghi nhận các trường hợp lỗi vào hệ thống MLOps/MLflow
                _log_validation_failure(func_name=func_name, errors=error_details)
                
                # Ném lỗi qua hệ thống exception tùy chỉnh để dừng pipeline an toàn
                raise DataValidationError(
                    message=f"Kiểm định dữ liệu thất bại tại hàm '{func_name}'.",
                    errors=error_details
                ) from e
                
            except Exception as unhandled_exception:
                logger.exception(f"[System Error] Lỗi không mong muốn khi kiểm định {func_name}")
                raise unhandled_exception

        return wrapper
    return decorator


def nutrition_check_io(head: int = 5, tail: int = 5, **kwargs):
    """
    Decorator kiểm định tường minh Input/Output sử dụng DataFrameSchema.
    Cho phép áp dụng các tuỳ chọn backend execution nâng cao nếu cần.
    
    Args:
        head, tail: Số dòng lỗi mẫu muốn hiển thị log.
        **kwargs: Khai báo tham số đầu vào và đầu ra cần check (VD: df=schema_in, out=schema_out).
    """
    def decorator(func: Callable) -> Callable:
        pandera_decorator = pa.check_io(head=head, tail=tail, **kwargs)
        wrapped_func = pandera_decorator(func)

        @functools.wraps(func)
        def wrapper(*func_args, **func_kwargs):
            func_name = func.__name__
            logger.info(f"[I/O Validation Start] Kiểm tra Input/Output cho: {func_name}")
            
            try:
                result = wrapped_func(*func_args, **func_kwargs)
                logger.info(f"[I/O Validation Success] Khớp Schema cho: {func_name}")
                return result
                
            except (SchemaError, SchemaErrors) as e:
                logger.error(f"[I/O Validation Failed] Lỗi cấu trúc I/O tại: {func_name}")
                
                error_details = getattr(PanderaErrorHandler, "extract_errors")(e)
                _log_validation_failure(func_name=func_name, errors=error_details)
                
                raise DataValidationError(
                    message=f"Lỗi kiểm định I/O tại hàm '{func_name}'.",
                    errors=error_details
                ) from e
                
        return wrapper
    return decorator