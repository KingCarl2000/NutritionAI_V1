"""
Module thực thi Backend kiểm định dành riêng cho Pandas DataFrame trong NutritionAI_V1.
Nhiệm vụ: Phân giải Schema, duyệt dữ liệu, tiền xử lý, thực thi vectorized checks và gom lỗi.
"""
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
import pandera as pa
from pandera.errors import SchemaErrors
from pandera.engines import pandas_engine

# Import logger từ hệ thống
from src.nutrition_core.logging.logger import logger
# Import DataValidationError để chuẩn hóa đầu ra lỗi
from src.nutrition_core.exception.val_exception import DataValidationError


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

    @classmethod
    def resolve_backend(cls, data: Any) -> bool:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Dữ liệu đầu vào phải là pandas.DataFrame, nhận được: {type(data)}")
        return True

    def _preprocess_and_coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        
        for col_name, col_schema in self.schema.columns.items():
            matched_columns = [
                c for c in df_processed.columns 
                if (re.match(col_name, c) if col_schema.regex else c == col_name)
            ]

            if not matched_columns:
                if col_schema.required:
                    self._record_error("Schema", f"Thiếu cột bắt buộc: {col_name}")
                continue

            if col_schema.coerce:
                for matched_col in matched_columns:
                    try:
                        expected_dtype = pandas_engine.Engine.dtype(col_schema.dtype)
                        df_processed[matched_col] = expected_dtype.coerce(df_processed[matched_col])
                    except Exception as e:
                        self._record_error(matched_col, f"Lỗi ép kiểu: {e}")

        return df_processed

    def _execute_vectorized_checks(self, df: pd.DataFrame) -> pd.Series:
        try:
            self.schema.validate(df, lazy=self.lazy)
            return pd.Series(True, index=df.index)
        except SchemaErrors as err:
            for _, row in err.failure_cases.iterrows():
                self._record_error(row.get('column', 'N/A'), row.get('failure_case', 'N/A'), row.get('index'))
            
            valid_mask = pd.Series(True, index=df.index)
            if 'index' in err.failure_cases:
                invalid_indices = err.failure_cases['index'].dropna().unique()
                valid_mask.loc[invalid_indices] = False
            return valid_mask

    def _record_error(self, column: str, reason: Any, index: Any = None):
        self.failure_cases.append({"column": column, "index": index, "reason": str(reason)})

    def run_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        self.resolve_backend(df)
        
        logger.info("[PandasBackend] Bắt đầu tiền xử lý và ép kiểu...")
        df_preprocessed = self._preprocess_and_coerce(df)

        logger.info("[PandasBackend] Thực thi các hàm Check Vector hóa...")
        valid_mask = self._execute_vectorized_checks(df_preprocessed)

        if self.drop_invalid_rows and not valid_mask.all():
            dropped_count = (~valid_mask).sum()
            logger.warning(f"[PandasBackend] Tự động loại bỏ {dropped_count} dòng vi phạm.")
            df_final = df_preprocessed[valid_mask].copy()
        else:
            df_final = df_preprocessed

        # Thay vì dùng SchemaError thuần, sử dụng DataValidationError để đồng bộ với logger decorator
        if self.failure_cases and not self.lazy:
            raise DataValidationError(
                message=f"Kiểm định thất bại nhanh (lazy=False) tại cột '{self.failure_cases[0]['column']}'.",
                errors=self.failure_cases
            )

        return df_final, self.failure_cases

def validate_with_pandas_backend(schema: pa.DataFrameSchema, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Wrapper API tiện ích cho các module khác gọi độc lập."""
    backend = NutritionPandasBackend(schema, **kwargs)
    valid_df, errors = backend.run_pipeline(df)
    
    if errors and not backend.drop_invalid_rows:
        logger.error(f"[PandasBackend] Tích lũy {len(errors)} lỗi trong quá trình xử lý.")
        raise DataValidationError("Quá trình kiểm định dữ liệu độc lập thất bại.", errors=errors)
        
    return valid_df