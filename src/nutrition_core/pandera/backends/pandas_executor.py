"""
Module thực thi Backend kiểm định dành riêng cho Pandas DataFrame trong NutritionAI_V1.
Nhiệm vụ: Phân giải Schema, duyệt dữ liệu, tiền xử lý, thực thi vectorized checks và gom lỗi.
"""
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import pandera as pa
from pandera.errors import SchemaError, SchemaErrors
from pandera.engines import pandas_engine

# Import logger từ hệ thống hiện tại của dự án
from src.nutrition_core.logging.logger import logger


class NutritionPandasBackend:
    """
    Backend chuyên biệt xử lý dữ liệu Pandas DataFrame.
    Hoạt động như một lớp trung gian (Facade) nhận cấu hình từ API BaseSchema 
    và điều phối quá trình kiểm định.
    """

    def __init__(self, schema: pa.DataFrameSchema, lazy: bool = True, drop_invalid_rows: bool = False):
        self.schema = schema
        self.lazy = lazy
        self.drop_invalid_rows = drop_invalid_rows
        self.failure_cases: List[Dict[str, Any]] = []

    # ==============================================================================
    # 1. Routing & Resolution
    # ==============================================================================
    @classmethod
    def resolve_backend(cls, data: Any) -> bool:
        """Đảm bảo Backend này chỉ kích hoạt khi dữ liệu đầu vào là Pandas DataFrame."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Dữ liệu đầu vào phải là pandas.DataFrame, nhận được: {type(data)}")
        return True

    # ==============================================================================
    # 2 & 3. Iteration & Preprocessing
    # ==============================================================================
    def _preprocess_and_coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Duyệt cấu trúc (Iteration) và Tiền xử lý (Preprocessing).
        Kiểm tra Meta-data, xử lý Regex, loại bỏ Null tùy chọn và kích hoạt Ép kiểu (Coercion).
        """
        df_processed = df.copy()
        
        # Duyệt qua các cột được định nghĩa trong Schema
        for col_name, col_schema in self.schema.columns.items():
            # Xử lý Regex matching cho tên cột
            matched_columns = [
                c for c in df_processed.columns 
                if (re.match(col_name, c) if col_schema.regex else c == col_name)
            ]

            # Bỏ qua cột nếu required=False và không tồn tại trong DF
            if not matched_columns:
                if col_schema.required:
                    self._record_error("Schema", f"Thiếu cột bắt buộc: {col_name}")
                continue

            # Ép kiểu dữ liệu qua pandas_engine nếu coerce=True
            if col_schema.coerce:
                for matched_col in matched_columns:
                    try:
                        expected_dtype = pandas_engine.Engine.dtype(col_schema.dtype)
                        df_processed[matched_col] = expected_dtype.coerce(df_processed[matched_col])
                    except Exception as e:
                        self._record_error(matched_col, f"Lỗi ép kiểu: {e}")

        return df_processed

    # ==============================================================================
    # 4. Native Execution (Thực thi mức dữ liệu)
    # ==============================================================================
    def _execute_vectorized_checks(self, df: pd.DataFrame) -> pd.Series:
        """
        Giao phó quá trình thực thi hàm Check cho core của Pandera để tận dụng API C/NumPy.
        """
        # Pandera mặc định đã tối ưu vectorized check với engine pandas
        try:
            # Tạm thời tắt lazy tại cấp độ pa.validate nếu đang muốn tự kiểm soát luồng filter
            validated_df = self.schema.validate(df, lazy=self.lazy)
            # Nếu không có lỗi, mảng boolean giả định là True toàn bộ
            return pd.Series(True, index=df.index)
        except SchemaErrors as err:
            # Chụp lại các lỗi để xử lý trong phần Result Handling
            for _, row in err.failure_cases.iterrows():
                self._record_error(row.get('column', 'N/A'), row.get('failure_case', 'N/A'), row.get('index'))
            
            # Khởi tạo mảng boolean đánh dấu các index lỗi là False
            valid_mask = pd.Series(True, index=df.index)
            if 'index' in err.failure_cases:
                invalid_indices = err.failure_cases['index'].dropna().unique()
                valid_mask.loc[invalid_indices] = False
            return valid_mask

    # ==============================================================================
    # 5. Result Handling & Error Aggregation
    # ==============================================================================
    def _record_error(self, column: str, reason: Any, index: Any = None):
        """Ghi nhận lỗi vào danh sách (dành cho chế độ Lazy)."""
        self.failure_cases.append({"column": column, "index": index, "reason": str(reason)})

    def run_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Hàm chính điều phối toàn bộ luồng duyệt, thực thi và tổng hợp kết quả.
        """
        self.resolve_backend(df)
        
        logger.info("[PandasBackend] Bắt đầu tiền xử lý và ép kiểu...")
        df_preprocessed = self._preprocess_and_coerce(df)

        logger.info("[PandasBackend] Thực thi các hàm Check Vector hóa...")
        valid_mask = self._execute_vectorized_checks(df_preprocessed)

        # Trích xuất và loại bỏ dòng lỗi nếu cấu hình drop_invalid_rows=True
        if self.drop_invalid_rows and not valid_mask.all():
            dropped_count = (~valid_mask).sum()
            logger.warning(f"[PandasBackend] Tự động loại bỏ {dropped_count} dòng vi phạm.")
            df_final = df_preprocessed[valid_mask].copy()
        else:
            df_final = df_preprocessed

        # Aggregation: Quản lý lỗi ném ra hoặc gom lại
        if self.failure_cases and not self.lazy:
            raise SchemaError(
                schema=self.schema,
                data=df_final,
                message=f"Kiểm định thất bại nhanh (lazy=False): {self.failure_cases[0]}"
            )

        return df_final, self.failure_cases

# Hỗ trợ tính năng tích hợp trực tiếp 
def validate_with_pandas_backend(schema: pa.DataFrameSchema, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Wrapper API cho Tầng BaseSchema gọi đến."""
    backend = NutritionPandasBackend(schema, **kwargs)
    valid_df, errors = backend.run_pipeline(df)
    
    if errors and not backend.drop_invalid_rows:
        logger.error(f"[PandasBackend] Tích lũy {len(errors)} lỗi trong quá trình xử lý.")
        # Hệ thống gọi tới có thể tự raise SchemaErrors dựa trên errors list
    return valid_df