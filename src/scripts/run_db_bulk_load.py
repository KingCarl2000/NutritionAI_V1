import os
import sys
import yaml
from pathlib import Path

# XÁC ĐỊNH ĐƯỜNG DẪN ROOT VÀ ĐƯA VÀO SYS.PATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nutrition_core.logging.logger import logger
from src.postgres.data_io.bulk_loader import PostgresBulkLoader

def main():
    logger.info("🚀 Khởi động tiến trình Bulk Load (Đọc từ tables_created_report.yaml)...")
    
    # 1. Đọc báo cáo danh sách bảng đã tạo thay vì schema.yaml
    REPORT_PATH = PROJECT_ROOT / "docs" / "architecture" / "tables_created_report.yaml"
    if not REPORT_PATH.exists():
        logger.error(f"❌ Không tìm thấy file báo cáo tại: {REPORT_PATH}")
        logger.error("Vui lòng chạy script 'create_tables.py' trước để sinh file báo cáo này!")
        return
        
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        created_report = yaml.safe_load(f)

    if not created_report:
        logger.warning("⚠️ File báo cáo tables_created_report.yaml đang trống.")
        return

    loader = PostgresBulkLoader(connection=None)
    data_raw_dir = PROJECT_ROOT / "Data" / "raw"

    # 2. Duyệt qua từng schema và các bảng thực tế từ báo cáo
    for schema_name, tables in created_report.items():
        logger.info(f"\n📂 Đang xử lý schema: '{schema_name}'")
        
        for final_table_name, table_meta in tables.items():
            source_dataset = table_meta.get("source_dataset")
            
            if not source_dataset:
                logger.warning(f"  ⚠️ Bảng '{final_table_name}' thiếu thông tin source_dataset. Bỏ qua.")
                continue

            # Xác định tên file CSV gốc trên ổ cứng
            # Ví dụ: Nếu final_table_name có prefix, ta tìm file tương ứng trong folder source_dataset
            # Hoặc bóc tách base name bằng cách loại bỏ prefix của dataset
            clean_dataset_prefix = source_dataset.lower().replace(" ", "_").replace("-", "_")
            # Tùy biến logic tìm file: Thử tìm trực tiếp file trùng với tên bảng gốc hoặc base name
            
            target_dir = data_raw_dir / source_dataset
            
            # Tìm file CSV dựa vào tên bảng gốc (nếu đã cắt bỏ prefix) hoặc dò tất cả các file trong thư mục
            csv_file_path = None
            if target_dir.exists():
                for csv_file in target_dir.glob("*.csv"):
                    # Kiểm tra xem file CSV này có khớp với tên base table không
                    base_name = csv_file.stem
                    if final_table_name.endswith(base_name) or final_table_name == base_name:
                        csv_file_path = csv_file
                        break
            
            if not csv_file_path or not csv_file_path.exists():
                logger.warning(f"  ⚠️ Bỏ qua bảng '{final_table_name}': Không tìm thấy file CSV tương ứng trong {target_dir}")
                continue

            target_table = f"{schema_name}.{final_table_name}"
            logger.info(f"  -> Đang nạp file '{csv_file_path.name}' vào bảng '{target_table}'...")
            
            try:
                loader.execute_bulk_load(
                    table_name=target_table,
                    file_path_on_server=str(csv_file_path),
                    delimiter=",",
                    header=True
                )
                logger.info(f"  ✅ Thành công: '{target_table}'")
            except Exception as e:
                logger.error(f"  ❌ Lỗi khi nạp '{target_table}': {e}")

    logger.info("\n🎉 Đã hoàn tất toàn bộ tiến trình Bulk Load!")

if __name__ == "__main__":
    main()