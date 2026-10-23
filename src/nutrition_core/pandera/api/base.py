"""
Module base định nghĩa các lớp cơ sở (Base Schemas) và tiện ích kiểm định dữ liệu dùng chung (Generic Validation Rules)
sử dụng Pandera API cho hệ thống NutritionAI_V1.
"""
from typing import Any, Dict, List, Optional, Type, TypeVar, Tuple
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

# Import custom Logger và MLOpsTracker từ thư mục logging của dự án
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker

# Import Backend tùy chỉnh từ tầng backends
from src.nutrition_core.pandera.backends.pandas_executor import NutritionPandasBackend

# Khởi tạo MLOpsTracker quản lý tiến trình Data Validation
mlflow_tracker = MLOpsTracker(experiment_name="Data_Validation_Pipeline")

# Type hint chung cho các Class kế thừa từ pa.DataFrameModel
SchemaModelT = TypeVar("SchemaModelT", bound=pa.DataFrameModel)


class BaseSchema(pa.DataFrameModel):
    """
    Lớp Schema cơ sở áp dụng các quy tắc kiểm định khung cho tất cả mô hình dữ liệu về Nutrition.
    Mọi Schema lớp con (như CaloriesSchema, UserSchema,...) đều nên kế thừa lớp này.
    """
    class Config:
        coerce = True                 # [Quy tắc 2] Tự động ép kiểu dữ liệu về đúng dtype khai báo
        strict = False                # [Quy tắc 1] Linh hoạt chấp nhận các cột phụ ở tầng Base
        ordered = False               # [Quy tắc 1] Không bắt buộc thứ tự cột trừ khi pipeline cụ thể yêu cầu
        unique_column_names = True    # [Quy tắc 1] Đảm bảo tên các cột trong DataFrame không bị trùng lặp
        add_missing_columns = True    # [Quy tắc 3] Tự động tạo cột và điền default value nếu dữ liệu thiếu cột không bắt buộc

    @classmethod
    def validate_df(
        cls: Type[SchemaModelT], 
        df: pd.DataFrame, 
        lazy: bool = True,
        drop_invalid_rows: bool = False
    ) -> pd.DataFrame:
        """
        Thực thi kiểm định một Pandas DataFrame thông qua NutritionPandasBackend (Tách biệt logic duyệt và lọc dữ liệu).

        Args:
            df (pd.DataFrame): DataFrame đầu vào cần kiểm định.
            lazy (bool): Thu thập tất cả vi phạm trước khi báo lỗi (mặc định True).
            drop_invalid_rows (bool): Kích hoạt tự động loại bỏ các dòng bị lỗi để pipeline tiếp tục chạy với dữ liệu sạch.

        Returns:
            pd.DataFrame: DataFrame đã được kiểm định (và được lọc sạch nếu drop_invalid_rows=True).

        Raises:
            ValueError / SchemaError: Khi dữ liệu có lỗi và cấu hình hệ thống không cho phép bỏ qua.
        """
        schema_name = cls.__name__
        total_rows = len(df)
        logger.info(f"[Pandera Base] Bắt đầu kiểm định DataFrame ({total_rows} dòng) với Schema '{schema_name}'...")
        
        # 1. Chuyển đổi DataFrameModel (Class) sang cấu trúc DataFrameSchema (Object) để giao phó cho Backend
        schema_object = cls.to_schema()
        
        # 2. Khởi tạo Backend Orchestrator
        backend = NutritionPandasBackend(
            schema=schema_object, 
            lazy=lazy, 
            drop_invalid_rows=drop_invalid_rows
        )
        
        try:
            # 3. Thực thi Pipeline xử lý trên Backend
            validated_df, errors = backend.run_pipeline(df)
            
            # Đánh giá kết quả và báo cáo MLflow
            if errors:
                error_count = len(errors)
                if drop_invalid_rows:
                    logger.warning(f"[Pandera Base] Schema '{schema_name}': Đã loại bỏ dòng lỗi. Tổng vi phạm: {error_count}.")
                    mlflow_tracker.log_params_and_metrics(
                        params={f"validation_{schema_name}_status": "PARTIAL_SUCCESS", f"validation_{schema_name}_schema": schema_name},
                        metrics={f"{schema_name}_error_count": error_count, f"{schema_name}_dropped_rows": total_rows - len(validated_df)}
                    )
                else:
                    # Nếu có lỗi nhưng hệ thống KHÔNG được chỉ định loại bỏ dòng, ngắt pipeline
                    logger.error(f"[Pandera Base] Kiểm định THẤT BẠI cho Schema '{schema_name}' với {error_count} vi phạm.")
                    mlflow_tracker.log_params_and_metrics(
                        params={f"validation_{schema_name}_status": "FAILED", f"validation_{schema_name}_schema": schema_name},
                        metrics={f"{schema_name}_error_count": error_count}
                    )
                    raise ValueError(f"Quá trình kiểm định dữ liệu trên Schema '{schema_name}' thất bại với {error_count} lỗi. Chi tiết: {errors[:5]}...")
            else:
                logger.info(f"[Pandera Base] Kiểm định dữ liệu THÀNH CÔNG toàn vẹn cho Schema '{schema_name}'.")
                mlflow_tracker.log_params_and_metrics(
                    params={f"validation_{schema_name}_status": "SUCCESS", f"validation_{schema_name}_schema": schema_name},
                    metrics={f"{schema_name}_error_count": 0}
                )
                
            return validated_df

        except SchemaError as e:
            # Bắt lỗi Fast-fail (Khi lazy=False ở tầng Backend)
            logger.error(f"[Pandera Base] Fast-fail kiểm định cho Schema '{schema_name}': {str(e)}")
            mlflow_tracker.log_params_and_metrics(
                params={f"validation_{schema_name}_status": "CRITICAL_FAILED", f"validation_{schema_name}_schema": schema_name}
            )
            raise