import os
import sys

# Đảm bảo import được các module từ thư mục src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.postgres.data_io.bulk_loader import PostgresBulkLoader
from src.postgres.core.db_config import config  # Import cấu hình DB đã thiết lập
from src.nutrition_core.exception import exception # Import xử lý ngoại lệ
from src.nutrition_core.logging.logger import logger           # Import module logger



# Dictionary cấu hình các dataset cần nạp
# Key: Tên bảng trong Database
# Value: Đường dẫn tuyệt đối đến file CSV trên Server
DATASETS_TO_LOAD = {
    "nutrition_data_raw": r"D:\NutritionAI_V1\Data\raw\health_data.csv",
    # Bạn có thể thêm các dataset khác trong tương lai ở đây:
    # "user_profiles_raw": r"D:\NutritionAI_V1\Data\raw\user_profiles.csv",
    # "food_ingredients_raw": r"D:\NutritionAI_V1\Data\raw\ingredients.csv"
}

def main():
    # 1. Lấy thông số kết nối từ db_config.py
    db_params = {
        "dbname": config.dbname,
        "user": config.user,
        "password": config.password,
        "host": config.host,
        "port": str(config.port)
    }

    # 2. Khởi tạo Bulk Loader
    loader = PostgresBulkLoader(db_params)

    # 3. Chạy vòng lặp qua dictionary để nạp toàn bộ file
    try:
        for table_name, file_path in DATASETS_TO_LOAD.items():
            logger.info(f"=== Bắt đầu tiến trình Bulk Load cho bảng '{table_name}' từ file '{file_path}' ===")
            
            loader.execute_bulk_load(
                table_name=table_name,
                file_path_on_server=file_path,
                format='csv',
                delimiter=',',
                header=True,
                temp_maintenance_work_mem=config.maintenance_work_mem # Sử dụng cấu hình tối ưu từ db_config
            )
            
            logger.info(f"=== Hoàn tất nạp dữ liệu cho bảng '{table_name}'! ===")
            
    except Exception as e:
        # Bắt và xử lý lỗi bằng class CustomException của dự án
        error_msg = f"Tiến trình nạp dữ liệu bị gián đoạn: {e}"
        logger.error(error_msg)
        raise exception.NutritionBaseException(error_msg, sys)
    finally:
        loader.close()
        logger.info("Đã đóng kết nối Database.")

if __name__ == "__main__":
    main()