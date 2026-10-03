import sys
import re
import hashlib
import time
import unicodedata
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, List, Union

from src.nutrition_core.logging.logger import logger, monitor_performance
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker
from src.nutrition_core.exception.exception import DataTransformationError


class BaseTransformer(ABC):
    """
    Lớp cơ sở trừu tượng cho tất cả các tác vụ Data Transformation trong Nutrition AI.
    Cung cấp các công cụ làm sạch, chuẩn hóa và bảo mật dữ liệu quy chuẩn,
    tích hợp theo dõi hiệu năng qua MLflow.
    """
    
    def __init__(self, mlflow_tracker: Optional[MLOpsTracker] = None):
        """
        Args:
            mlflow_tracker: Đối tượng theo dõi MLflow để ghi nhận metrics và lỗi.
        """
        self.mlflow_tracker = mlflow_tracker

    def fit(self, X: pd.DataFrame, y=None):
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @monitor_performance(time_limit=10.0)
    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Thực thi tuần tự fit và transform, đo lường thời gian và đồng bộ lên MLflow.
        """
        step_name = self.__class__.__name__
        logger.info(f"Bắt đầu thực thi fit_transform cho {step_name}...")
        start_time = time.time()
        
        try:
            result_df = self.fit(df, y).transform(df)
            
            # Ghi nhận thời gian thực thi lên MLflow Tracking
            exec_time = time.time() - start_time
            if self.mlflow_tracker:
                self.mlflow_tracker.log_params_and_metrics(
                    metrics={f"{step_name}_exec_time_seconds": exec_time}
                )
                
            return result_df
        except Exception as e:
            error_msg = f"Lỗi trong quá trình fit_transform của {step_name}: {e}"
            logger.error(error_msg)
            if self.mlflow_tracker:
                self.mlflow_tracker.log_error_and_tag(error_msg=error_msg)
            raise DataTransformationError("Thất bại khi thực thi fit_transform", error_details=sys)

    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            new_cols = []
            for col in df.columns:
                normalized_str = unicodedata.normalize('NFKD', str(col)).encode('ASCII', 'ignore').decode('utf-8')
                snake_case_str = re.sub(r'[^a-z0-9]+', '_', normalized_str.strip().lower()).strip('_')
                new_cols.append(snake_case_str)
                
            df.columns = new_cols
            logger.debug("Đã chuẩn hóa tên cột về định dạng snake_case.")
            return df
        except Exception as e:
            raise DataTransformationError(f"Lỗi khi chuẩn hóa tên cột: {e}", error_details=sys)

    def trim_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            str_cols = df.select_dtypes(include=['object', 'string']).columns
            for col in str_cols:
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            logger.debug(f"Đã cắt khoảng trắng thừa cho {len(str_cols)} cột chuỗi.")
            return df
        except Exception as e:
            raise DataTransformationError(f"Lỗi khi trim khoảng trắng: {e}", error_details=sys)

    def handle_empty_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            str_cols = df.select_dtypes(include=['object', 'string']).columns
            for col in str_cols:
                df[col] = df[col].replace(r'^\s*$', pd.NA, regex=True)
            logger.debug("Đã thay thế chuỗi rỗng bằng pd.NA.")
            return df
        except Exception as e:
            raise DataTransformationError(f"Lỗi khi xử lý chuỗi rỗng: {e}", error_details=sys)

    def drop_duplicates_by_key(self, df: pd.DataFrame, subset: Union[str, List[str]], sort_col: Optional[str] = None, keep: str = 'last') -> pd.DataFrame:
        try:
            initial_len = len(df)
            if sort_col:
                if sort_col not in df.columns:
                    logger.warning(f"Cột sắp xếp '{sort_col}' không tồn tại. Bỏ qua bước sắp xếp.")
                else:
                    df = df.sort_values(by=sort_col)
                    
            df = df.drop_duplicates(subset=subset, keep=keep)
            final_len = len(df)
            dropped_count = initial_len - final_len
            
            if dropped_count > 0:
                logger.info(f"Đã loại bỏ {dropped_count} bản ghi trùng lặp dựa trên khóa {subset}.")
                # Đẩy metric số lượng dòng bị xóa lên MLflow
                if self.mlflow_tracker:
                    self.mlflow_tracker.log_params_and_metrics(
                        metrics={f"{self.__class__.__name__}_dropped_duplicates": dropped_count}
                    )
            return df
        except Exception as e:
            raise DataTransformationError(f"Lỗi khi khử trùng lặp dữ liệu: {e}", error_details=sys)

    def hash_pii_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
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