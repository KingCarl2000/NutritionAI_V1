import sys
import re
import hashlib
import unicodedata
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, List, Union

# Tái sử dụng các module core từ NutritionAI_V1
from src.nutrition_core.logging.logger import logger, monitor_performance
from src.nutrition_core.exception.exception import DataTransformationError


class BaseTransformer(ABC):
    """
    Lớp cơ sở trừu tượng cho tất cả các tác vụ Data Transformation.
    Cung cấp các công cụ làm sạch, chuẩn hóa và bảo mật dữ liệu quy chuẩn.
    """
    
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        """
        Lưu lại các trạng thái của tập train nếu cần (ví dụ: mean, median).
        Mặc định không thực hiện thay đổi nào.
        """
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hàm bắt buộc phải ghi đè ở các pipeline con để thực hiện logic nghiệp vụ đặc thù.
        """
        pass

    @monitor_performance(time_limit=10.0)
    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Thực thi tuần tự fit và transform, có tích hợp giám sát hiệu năng.
        """
        logger.info("Bắt đầu thực thi fit_transform...")
        try:
            return self.fit(df, y).transform(df)
        except Exception as e:
            logger.error(f"Lỗi trong quá trình fit_transform: {e}")
            raise DataTransformationError("Thất bại khi thực thi fit_transform", error_details=sys)

    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn hóa tên cột: loại bỏ dấu tiếng Việt, chuyển sang snake_case, xóa ký tự đặc biệt.
        """
        try:
            new_cols = []
            for col in df.columns:
                # Loại bỏ dấu tiếng Việt
                normalized_str = unicodedata.normalize('NFKD', str(col)).encode('ASCII', 'ignore').decode('utf-8')
                # Chuyển chữ thường và thay thế ký tự không phải chữ/số bằng '_'
                snake_case_str = re.sub(r'[^a-z0-9]+', '_', normalized_str.strip().lower())
                # Xóa dấu '_' thừa ở đầu và cuối
                snake_case_str = snake_case_str.strip('_')
                new_cols.append(snake_case_str)
                
            df.columns = new_cols
            logger.debug("Đã chuẩn hóa tên cột về định dạng snake_case.")
            return df
        except Exception as e:
            raise DataTransformationError(f"Lỗi khi chuẩn hóa tên cột: {e}", error_details=sys)

    def trim_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cắt bỏ khoảng trắng thừa ở đầu và cuối cho các cột chứa chuỗi.
        """
        try:
            str_cols = df.select_dtypes(include=['object', 'string']).columns
            for col in str_cols:
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            logger.debug(f"Đã cắt khoảng trắng thừa cho {len(str_cols)} cột chuỗi.")
            return df
        except Exception as e:
            raise DataTransformationError(f"Lỗi khi trim khoảng trắng: {e}", error_details=sys)

    def handle_empty_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thay thế các chuỗi rỗng hoàn toàn hoặc chỉ chứa khoảng trắng thành pd.NA.
        """
        try:
            str_cols = df.select_dtypes(include=['object', 'string']).columns
            for col in str_cols:
                # Sử dụng regex để tìm các chuỗi chỉ chứa khoảng trắng hoặc rỗng
                df[col] = df[col].replace(r'^\s*$', pd.NA, regex=True)
            logger.debug("Đã thay thế chuỗi rỗng bằng pd.NA.")
            return df
        except Exception as e:
            raise DataTransformationError(f"Lỗi khi xử lý chuỗi rỗng: {e}", error_details=sys)

    def drop_duplicates_by_key(self, df: pd.DataFrame, subset: Union[str, List[str]], sort_col: Optional[str] = None, keep: str = 'last') -> pd.DataFrame:
        """
        Khử trùng lặp dựa trên Primary Key. Có thể sắp xếp theo thời gian để giữ bản ghi mới nhất.
        """
        try:
            initial_len = len(df)
            if sort_col:
                if sort_col not in df.columns:
                    logger.warning(f"Cột sắp xếp '{sort_col}' không tồn tại. Bỏ qua bước sắp xếp.")
                else:
                    df = df.sort_values(by=sort_col)
                    
            df = df.drop_duplicates(subset=subset, keep=keep)
            final_len = len(df)
            
            if initial_len != final_len:
                logger.info(f"Đã loại bỏ {initial_len - final_len} bản ghi trùng lặp dựa trên khóa {subset}.")
            return df
        except Exception as e:
            raise DataTransformationError(f"Lỗi khi khử trùng lặp dữ liệu: {e}", error_details=sys)

    def hash_pii_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Băm SHA-256 các cột chứa dữ liệu nhạy cảm (PII) để ẩn danh.
        """
        try:
            for col in columns:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: hashlib.sha256(str(x).encode('utf-8')).hexdigest() if pd.notna(x) else x
                    )
                else:
                    logger.warning(f"Cột PII '{col}' không tồn tại trong DataFrame để băm.")
            
            logger.info(f"Đã băm SHA-256 ẩn danh cho các cột: {columns}")
            return df
        except Exception as e:
            raise DataTransformationError(f"Lỗi khi mã hóa dữ liệu PII: {e}", error_details=sys)