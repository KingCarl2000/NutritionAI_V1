import os
import sys
import glob
from pathlib import Path
from typing import Dict, Any

# XÁC ĐỊNH ĐƯỜNG DẪN ROOT VÀ ĐƯA VÀO SYS.PATH
# Sửa lại thành absolute path linh hoạt
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import các core modules của bạn
from src.nutrition_core.logging.logger import logger
from src.postgres.core.db_config import config
from src.postgres.data_io.bulk_loader import PostgresBulkLoader
from src.postgres.core.connection import get_connection  # Import pool connection của bạn

# ĐỊNH NGHĨA DANH SÁCH DATASET
DATASETS_TO_LOAD: Dict[str, Dict[str, Any]] = {
    # 'format' giờ đây hỗ trợ 'folder' (quét tất cả .csv) hoặc 'file' (nạp 1 file duy nhất)
    # Tên key "raw.fitness_tracker_dataset" chỉ mang tính đại diện khi dùng mode 'folder',
    # các table_name thực tế sẽ lấy từ TÊN FILE CSV (ví dụ: raw.dailyActivity_merged)
    "raw": {
        "file_path": str(PROJECT_ROOT / "Data" / "raw" / "Fitabase Data 3.12.16-4.11.16"),
        "format": "folder",
        "delimiter": ",",
        "header": True
    }
}

def process_file_load(loader: PostgresBulkLoader, schema_name: str, table_name: str, csv_file_path: str, dataset_config: dict):
    """Hàm phụ trợ để thực thi bulk load cho một file duy nhất"""
    target_table = f"{schema_name}.{table_name}"
    
    logger.info(f"  -> Đang nạp file '{Path(csv_file_path).name}' vào bảng '{target_table}'...")
    
    try:
        loader.execute_bulk_load(
            table_name=target_table,
            file_path_on_server=csv_file_path,
            delimiter=dataset_config.get("delimiter", ","),
            header=dataset_config.get("header", True)
        )
        logger.info(f"  ✅ Thành công: '{target_table}'")
    except Exception as e:
        logger.error(f"  ❌ Lỗi khi nạp '{target_table}': {e}")
        # Tuỳ vào logic dự án, bạn có thể raise e để dừng toàn bộ, 
        # hoặc chỉ ghi log và chạy tiếp file khác. Ở đây tôi chọn log và tiếp tục.

def main():
    logger.info("🚀 Khởi động tiến trình Bulk Load dữ liệu...")
    
    # Sử dụng Connection từ Pool bằng context manager (with)
    # Đảm bảo connection tự động trả về pool sau khi xong việc
    with get_connection() as conn:
        # Khởi tạo loader và truyền connection trực tiếp
        loader = PostgresBulkLoader(connection=conn)
        
        # Duyệt qua các cấu hình dataset
        for config_key, dataset_config in DATASETS_TO_LOAD.items():
            if "." in config_key:
                schema_name, base_table = config_key.split(".", 1)
            else:
                schema_name = config_key  
                base_table = config_key
                
            target_path = dataset_config["file_path"]
            data_format = dataset_config.get("format", "file")
            
            logger.info(f"\n=== Bắt đầu xử lý cấu hình: '{config_key}' (Format: {data_format}) ===")
            
            if not os.path.exists(target_path):
                logger.error(f"❌ Không tìm thấy đường dẫn: {target_path}")
                continue

            # TRƯỜNG HỢP 1: Xử lý Folder
            if data_format.lower() == "folder":
                if not os.path.isdir(target_path):
                    logger.error(f"❌ Cấu hình là 'folder' nhưng đường dẫn lại là file: {target_path}")
                    continue
                    
                csv_files = glob.glob(os.path.join(target_path, "*.csv"))
                if not csv_files:
                    logger.warning(f"⚠️ Không tìm thấy file .csv nào trong thư mục: {target_path}")
                    continue
                    
                logger.info(f"📂 Đã tìm thấy {len(csv_files)} file CSV. Bắt đầu nạp hàng loạt...")
                for file_path in csv_files:
                    file_table_name = Path(file_path).stem
                    process_file_load(
                        loader=loader,
                        schema_name=schema_name,
                        table_name=file_table_name,
                        csv_file_path=file_path,
                        dataset_config=dataset_config
                    )
                    
            # TRƯỜNG HỢP 2: Xử lý 1 File duy nhất 
            elif data_format.lower() == "file":
                process_file_load(
                    loader=loader,
                    schema_name=schema_name,
                    table_name=base_table,
                    csv_file_path=target_path,
                    dataset_config=dataset_config
                )
            else:
                logger.error(f"❌ Định dạng 'format' không hợp lệ: {data_format}. Dùng 'folder' hoặc 'file'.")
                
        # Vì autocommit=False trong connection pool, ta cần commit sau khi xong toàn bộ
        conn.commit() 
        logger.info("\n🎉 Đã hoàn tất tiến trình Bulk Load và Commit dữ liệu!")

if __name__ == "__main__":
    main()