"""
Module định nghĩa các ngoại lệ (Exceptions) và công cụ xử lý lỗi 
cho quá trình kiểm định dữ liệu (Data Validation) bằng Pandera.
"""
from typing import Any, Dict, List, Union
import pandas as pd
from pandera.errors import SchemaError, SchemaErrors

class DataValidationError(Exception):
    """
    Ngoại lệ tùy chỉnh cấp cao nhất cho các lỗi kiểm định dữ liệu trong hệ thống NutritionAI.
    Bọc lại các lỗi của Pandera để chuẩn hóa đầu ra cho API hoặc Logging.
    """
    def __init__(self, message: str, errors: List[Dict[str, Any]] = None):
        super().__init__(message)
        self.errors = errors or []

class PanderaErrorHandler:
    """
    Tiện ích xử lý và trích xuất lỗi từ Pandera Errors (SchemaError, SchemaErrors).
    Tương ứng với Kịch bản 1 & 2 trong tài liệu của bạn.
    """
    
    @staticmethod
    def parse_lazy_errors(err: SchemaErrors) -> List[Dict[str, Any]]:
        """
        [Kịch bản 2: Lazy Validation]
        Trích xuất chi tiết dữ liệu lỗi từ DataFrame failure_cases của Pandera.
        """
        parsed_errors = []
        if isinstance(err.failure_cases, pd.DataFrame):
            for _, row in err.failure_cases.iterrows():
                parsed_errors.append({
                    "column": row.get("column", "N/A"),
                    "check_rule": row.get("check", "N/A"),
                    "failure_value": row.get("failure_case", "N/A"),
                    "index": row.get("index", None)
                })
        return parsed_errors

    @staticmethod
    def parse_eager_error(err: SchemaError) -> List[Dict[str, Any]]:
        """
        [Kịch bản 1: Eager Validation]
        Trích xuất lỗi dừng đột ngột đầu tiên.
        """
        return [{
            "column": getattr(err, "column", "N/A"),
            "check_rule": str(err.check),
            "failure_value": err.failure_cases.iloc[0] if isinstance(err.failure_cases, pd.Series) else err.failure_cases,
            "reason_code": str(getattr(err, 'reason_code', 'N/A'))
        }]

    @classmethod
    def handle_validation_error(cls, err: Union[SchemaError, SchemaErrors], schema_name: str) -> DataValidationError:
        """
        Hàm bao chuẩn hóa lỗi để ném ra DataValidationError.
        """
        if isinstance(err, SchemaErrors):
            details = cls.parse_lazy_errors(err)
            msg = f"[{schema_name}] Lazy Validation thất bại với {len(details)} lỗi."
        elif isinstance(err, SchemaError):
            details = cls.parse_eager_error(err)
            msg = f"[{schema_name}] Eager Validation thất bại tại cột '{getattr(err, 'column', 'N/A')}'."
        else:
            details = [{"error": str(err)}]
            msg = f"[{schema_name}] Lỗi kiểm định không xác định."

        return DataValidationError(message=msg, errors=details)