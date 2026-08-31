import os
import sys
from pathlib import Path
from typing import Dict, Any

# Xác định đường dẫn gốc project (NutritionAI_V1) và đưa vào sys.path để Python nhận diện package 'src'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nutrition_core.logging.logger import logger
from src.postgres.core.db_config import config
from src.postgres.data_io.bulk_loader import PostgresBulkLoader

# Định nghĩa danh sách các dataset sẽ load
DATASETS_TO_LOAD: Dict[str, Dict[str, Any]] = {
    "raw.fitness_tracker_dataset": {
        "file_path": str(PROJECT_ROOT / "Data" / "raw" / "fitness_tracker_dataset.csv"),
        "format": "csv",
        "delimiter": ",",
        "header": True
    }
}

def main():
    logger.info("Khởi động tiến trình Bulk Load dữ liệu...")

    # Lấy thông số kết nối từ db_config.py
    db_params = {
        "dbname": config.dbname,
        "user": config.user,
        "password": config.password,
        "host": config.host,
        "port": str(config.port)
    }
    
    # Khởi tạo loader kết nối DB với đúng tham số connection_params
    loader = PostgresBulkLoader(connection_params=db_params)
    
    for full_table_name, dataset_config in DATASETS_TO_LOAD.items():
        if "." in full_table_name:
            schema_name, table_name = full_table_name.split(".", 1)
        else:
            schema_name = "public"
            table_name = full_table_name

        file_path = dataset_config["file_path"]
        
        logger.info(f"=== Bắt đầu Bulk Load: Bảng '{schema_name}.{table_name}' | File '{file_path}' ===")
        
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu tại đường dẫn: {file_path}")

            # Gọi execute_bulk_load với định dạng bảng đầy đủ "schema.table" 
            # để tránh lỗi không nhận tham số schema_name riêng biệt
            target_table = f"{schema_name}.{table_name}"
            
            loader.execute_bulk_load(
                table_name=target_table,
                file_path_on_server=file_path,
                delimiter=dataset_config.get("delimiter", ","),
                header=dataset_config.get("header", True)
            )
            logger.info(f"✅ Nạp dữ liệu thành công cho bảng '{schema_name}.{table_name}'")
        except Exception as e:
            logger.error(f"❌ Lỗi khi nạp dữ liệu bảng '{schema_name}.{table_name}': {e}")
            raise e

if __name__ == "__main__":
    main()
