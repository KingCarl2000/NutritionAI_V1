import sys
import decimal
import numpy as np
import pandas as pd
import psycopg
# Thêm thư viện này của SQLAlchemy
from sqlalchemy import text 

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
# ... (giữ nguyên các import khác)

from src.nutrition_core.logging.logger import logger, monitor_performance
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker
from src.nutrition_core.exception.exception import (
    DatabaseException,
    DatabaseConnectionError,
    QueryExecutionError,
    DataTransformationError
)

class BasePostgresExtractor(ABC):
    """
    Lớp cơ sở quản lý kết nối và trích xuất dữ liệu từ PostgreSQL.
    Bao gồm đóng gói bản ghi (Record Packaging) và ánh xạ kiểu dữ liệu (Data Type Mapping).
    """
    
    def __init__(self):
        pass

    def _get_connection(self):
        try:
            from src.postgres.core.connection import get_connection
            return get_connection()
        except Exception as e:
            logger.error(f"Lỗi khi import hoặc khởi tạo kết nối database: {e}")
            raise DatabaseConnectionError("Không thể lấy kết nối PostgreSQL từ Pool.", error_details=sys)

    def _convert_data_types(
        self, 
        df: pd.DataFrame, 
        schema_mapping: Optional[Dict[str, Any]] = None,
        flatten_json_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Ánh xạ kiểu dữ liệu từ PostgreSQL sang Pandas/NumPy DataFrame.
        """
        try:
            # 1. Xử lý JSON/JSONB: Phẳng hóa (flatten) các trường lồng nhau
            if flatten_json_cols:
                for col in flatten_json_cols:
                    if col in df.columns:
                        # Trích xuất các dict bên trong cột JSON thành các cột riêng biệt
                        flattened_df = pd.json_normalize(df[col])
                        # Thêm tiền tố để tránh trùng lặp tên cột
                        flattened_df.columns = [f"{col}_{subcol}" for subcol in flattened_df.columns]
                        df = pd.concat([df.drop(columns=[col]), flattened_df], axis=1)
                        logger.debug(f"Đã phẳng hóa cột JSON: '{col}'")

            # 2. Xử lý tự động các kiểu dữ liệu thô từ psycopg3
            for col in df.columns:
                # Lấy giá trị hợp lệ đầu tiên để nội suy kiểu dữ liệu thô
                first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                
                if first_valid is None:
                    continue

                # Numeric / Decimal -> float64
                if isinstance(first_valid, decimal.Decimal):
                    df[col] = df[col].astype('float64')
                    
                # Boolean -> Nullable Boolean (pd.BooleanDtype)
                elif isinstance(first_valid, bool):
                    df[col] = df[col].astype('boolean')
                    
                # Array (Mảng) -> NumPy Array (Hữu ích cho pgvector / embeddings)
                elif isinstance(first_valid, list):
                    values = [np.asarray(value) if isinstance(value, list) else value for value in df[col]]
                    df[col] = pd.Series(values, index=df.index, dtype=object)

            # 3. Xử lý ép kiểu tường minh (bao gồm Timestamp) qua schema_mapping
            if schema_mapping:
                for col, dtype in schema_mapping.items():
                    if col in df.columns:
                        if dtype in ['datetime64', 'datetime64[ns]']:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        else:
                            df[col] = df[col].astype(dtype)
                logger.info("Đã áp dụng ánh xạ kiểu dữ liệu tường minh.")

            return df
            
        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi định dạng dữ liệu (Type Conversion): {e}")
            raise DataTransformationError(
                "Thất bại khi ánh xạ kiểu dữ liệu PostgreSQL sang cấu trúc DataFrame.", 
                error_details=sys
            )

    @monitor_performance(time_limit=10.0)
    def fetch_as_dataframe(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None,
        schema_mapping: Optional[Dict[str, Any]] = None,
        flatten_json_cols: Optional[List[str]] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        mlflow_tracker: Optional[MLOpsTracker] = None,
        dataset_name: str = "raw_postgresql_data"
    ) -> pd.DataFrame:
        
        params = params or {}
        logger.info("Bắt đầu truy vấn trích xuất dữ liệu từ PostgreSQL...")

        try:
            with self._get_connection() as conn:
                # SỬA LỖI: Dùng pd.read_sql với text() thay vì gọi conn.cursor()
                sql_query = text(query)
                df = pd.read_sql(sql_query, conn, params=params)
                
                if df.empty:
                    logger.warning("Truy vấn thực thi thành công nhưng kết quả trả về rỗng.")
                    return pd.DataFrame()
                
                logger.info(f"Đã trích xuất thành công {len(df)} bản ghi thô.")
                
                # Chuyển đổi và ánh xạ kiểu dữ liệu (Giữ nguyên logic cũ của bạn)
                df = self._convert_data_types(df, schema_mapping, flatten_json_cols)

                # Thiết lập Index nếu được chỉ định
                if index_col:
                    missing_cols = [col for col in (index_col if isinstance(index_col, list) else [index_col]) if col not in df.columns]
                    if not missing_cols:
                        df.set_index(index_col, inplace=True)
                        logger.debug(f"Đã thiết lập cột {index_col} làm DataFrame Index.")
                    else:
                        logger.warning(f"Không thể đặt Index. Cột {missing_cols} không tồn tại trong kết quả.")

                # GHI NHẬN DATA LINEAGE VÀO MLFLOW
                if mlflow_tracker is not None:
                    source_uri = f"postgresql://query_hash_{hash(query)}" 
                    mlflow_tracker.log_dataset_lineage(df, dataset_name, source_uri)
                    logger.info("Đã đồng bộ thông tin Dataset Lineage lên MLflow.")

                return df

        # Có thể bắt thêm SQLAlchemy Error nếu cần thiết
        except Exception as db_err:
            logger.error(f"Lỗi thực thi dữ liệu từ PostgreSQL: {db_err}")
            raise QueryExecutionError("Lỗi thực thi câu lệnh SQL qua SQLAlchemy.", error_details=sys)

    @abstractmethod
    def extract(self, *args, **kwargs) -> pd.DataFrame:
        pass