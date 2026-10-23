"""
Module cấu hình toàn cục cho Pandera trong hệ thống NutritionAI_V1.
Quản lý các mức độ kiểm định (ValidationDepth) dựa trên môi trường triển khai,
đồng thời tích hợp theo dõi hiệu năng và ghi log cấu hình lên MLflow.
"""

import os
import pandera as pa
from pandera.config import ValidationDepth, config_context
from typing import Optional

from src.nutrition_core.logging.logger import logger, monitor_performance
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker

@monitor_performance(time_limit=2.0)
def setup_pandera_validation(mlflow_tracker: Optional[MLOpsTracker] = None) -> config_context:
    """
    Khởi tạo mức độ kiểm định toàn cục từ biến môi trường.
    Được gọi một lần tại điểm khởi chạy ứng dụng (ví dụ: data validation pipeline).
    
    Args:
        mlflow_tracker: Instance của MLOpsTracker để ghi nhận tham số cấu hình (tùy chọn).
    """
    # Lấy cấu hình từ biến môi trường, mặc định là SCHEMA_AND_DATA cho môi trường dev/staging
    env_depth = os.getenv("PANDERA_VALIDATION_DEPTH", "SCHEMA_AND_DATA").upper()
    
    # Map giá trị môi trường sang Enum của Pandera
    depth_mapping = {
        "SCHEMA_ONLY": ValidationDepth.SCHEMA_ONLY,
        "DATA_ONLY": ValidationDepth.DATA_ONLY,
        "SCHEMA_AND_DATA": ValidationDepth.SCHEMA_AND_DATA
    }
    
    validation_depth = depth_mapping.get(env_depth, ValidationDepth.SCHEMA_AND_DATA)
    
    logger.info(f"Khởi tạo cấu hình Pandera. Mức độ kiểm định: {validation_depth.name}")
    
    # Ghi log tham số cấu hình lên MLflow nếu có tracker
    if mlflow_tracker:
        try:
            mlflow_tracker.log_params_and_metrics(params={
                "pandera_validation_depth": validation_depth.name
            })
            logger.info("Đã ghi log cấu hình Pandera lên MLflow.")
        except Exception as e:
            logger.error(f"Lỗi khi ghi log cấu hình Pandera lên MLflow: {e}")
            
    # Áp dụng cấu hình
    return config_context(validation_depth=validation_depth)