import os
import sys
import yaml
from pathlib import Path

# XÁC ĐỊNH ĐƯỜNG DẪN ROOT VÀ ĐƯA VÀO SYS.PATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nutrition_core.logging.logger import logger
from src.data_io.bulk_loader import PostgresBulkLoader

def main(config_rel_path: str):
    logger.info(f"🚀 Khởi động tiến trình Bulk Load từ cấu hình: {config_rel_path}")

    # 1. Xác định đường dẫn file config
    config_path = PROJECT_ROOT / config_rel_path
    
    if not config_path.exists():
        logger.error(f"❌ Không tìm thấy file config tại: {config_path}")
        return

    # 2. Đọc file config dataset
    with open(config_path, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)

    if not dataset_config:
        logger.warning(f"⚠️ File config {config_rel_path} đang trống.")
        return

    loader = PostgresBulkLoader(connection=None)

    # 3. Duyệt qua các dataset được khai báo trong file YAML (hỗ trợ đọc linh hoạt calories, recipes...)
    for dataset_key, dataset_info in dataset_config.items():
        logger.info(f"\n📂 Đang xử lý dataset định tuyến: '{dataset_key}'")
        
        # Tự động lấy các tham số thiết lập từ YAML
        table_name = dataset_info.get("table_name", dataset_key)
        schema_name = dataset_info.get("db_schema", "raw")
        source_path_str = dataset_info.get("source_path")

        if not source_path_str:
            logger.warning(f"  ⚠️ Dataset '{dataset_key}' thiếu thông tin 'source_path'. Bỏ qua.")
            continue

        # Xử lý đường dẫn linh hoạt (hỗ trợ cả đường dẫn tuyệt đối như 'D:/...' hoặc tương đối)
        csv_file_path = Path(source_path_str)
        if not csv_file_path.is_absolute():
            csv_file_path = PROJECT_ROOT / csv_file_path

        if not csv_file_path.exists():
            logger.warning(f"  ⚠️ Không tìm thấy file dữ liệu tại: {csv_file_path}")
            continue

        target_table = f"{schema_name}.{table_name}"
        logger.info(f"  -> Đang nạp file '{csv_file_path.name}' vào '{target_table}'...")

        try:
            # Gọi hàm execute_bulk_load từ class PostgresBulkLoader
            loader.execute_bulk_load(
                table_name=table_name,
                file_path_on_server=str(csv_file_path),
                schema_name=schema_name,
                delimiter=",",
                format_type="CSV",
                has_header=True
            )
            logger.info(f"  ✅ Thành công nạp dữ liệu vào {target_table}!")
        except Exception as e:
            logger.error(f"  ❌ Quá trình Bulk Load thất bại cho {target_table}: {e}")

if __name__ == "__main__":
    # Danh sách các file cấu hình dataset cần nạp.
    # Dễ dàng mở rộng cho các dataset khác bằng cách truyền thêm đường dẫn vào danh sách này.
    config_files = [
        "config/datasets/calories.yaml",
        # "config/datasets/recipes.yaml",  <-- Mở comment và thêm file cấu hình mới tại đây
        # "config/datasets/users.yaml"
    ]
    
    for cfg in config_files:
        main(cfg)