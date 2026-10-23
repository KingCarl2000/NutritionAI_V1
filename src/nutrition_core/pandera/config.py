"""
Module cấu hình toàn cục cho Pandera trong hệ thống NutritionAI_V1.
Quản lý các mức độ kiểm định (ValidationDepth) dựa trên môi trường triển khai.
"""
import os
import pandera as pa
from pandera.config import ValidationDepth, config_context
from src.nutrition_core.logging.logger import logger

def setup_pandera_validation():
    """
    Khởi tạo mức độ kiểm định toàn cục từ biến môi trường.
    Được gọi một lần tại điểm khởi chạy ứng dụng (ví dụ: src/api/main.py).
    
    Các mức độ hỗ trợ (PANDERA_VALIDATION_DEPTH):
    - SCHEMA_ONLY: Chỉ kiểm tra cấu trúc (nhanh, dùng cho lazy data/schema validation).
    - DATA_ONLY: Chỉ kiểm tra logic nội dung (bỏ qua dtype/cột).
    - SCHEMA_AND_DATA: Kiểm tra toàn diện (Mặc định).
    """
    val_depth_str = os.getenv("PANDERA_VALIDATION_DEPTH", "SCHEMA_AND_DATA").upper()

    depth_mapping = {
        "SCHEMA_ONLY": ValidationDepth.SCHEMA_ONLY,
        "DATA_ONLY": ValidationDepth.DATA_ONLY,
        "SCHEMA_AND_DATA": ValidationDepth.SCHEMA_AND_DATA
    }

    if val_depth_str not in depth_mapping:
        logger.warning(
            f"[Pandera Config] Giá trị '{val_depth_str}' không hợp lệ. "
            "Tự động Fallback về mặc định an toàn: SCHEMA_AND_DATA."
        )
        val_depth_str = "SCHEMA_AND_DATA"

    # Thiết lập cấu hình toàn cục
    pa.set_config(validation_depth=depth_mapping[val_depth_str])
    logger.info(f"[Pandera Config] Đã thiết lập validation_depth toàn cục ở mức: {val_depth_str}")


def override_validation_context(depth: str):
    """
    Context manager để tạm thời ghi đè độ sâu kiểm định cho một luồng pipeline cụ thể 
    mà không làm thay đổi cấu hình toàn cục.
    
    Args:
        depth (str): "SCHEMA_ONLY", "DATA_ONLY", hoặc "SCHEMA_AND_DATA"
        
    Example:
        with override_validation_context("SCHEMA_ONLY"):
            # Chạy kiểm tra nhanh cấu trúc cho file nháp
            validate_df(df_draft)
    """
    depth_upper = depth.upper()
    try:
        selected_depth = ValidationDepth[depth_upper]
        return config_context(validation_depth=selected_depth)
    except KeyError:
        raise ValueError(f"Độ sâu kiểm định '{depth}' không hợp lệ. Hãy chọn SCHEMA_ONLY, DATA_ONLY, hoặc SCHEMA_AND_DATA.")