import os
import json
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

from src.nutrition_core.logging.logger import logger
from src.data_io.base_extractor import BasePostgresExtractor
from src.data_io.dvc_handler import DVCHandler

class CaloriesDataIngestion(BasePostgresExtractor):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.dvc = DVCHandler()
        
    def _read_schema(self) -> list:
        schema_file = self.config['schema']['json_schema_file']
        logger.info(f"Đọc cấu hình JSON Schema từ: {schema_file}")
        try:
            with open(schema_file, 'r') as f:
                schema = json.load(f)
            return schema.get('columns_to_drop', [])
        except FileNotFoundError:
            logger.warning(f"Không tìm thấy file {schema_file}, bỏ qua việc drop columns.")
            return []

    def extract_data_from_postgresql(self) -> pd.DataFrame:
        schema_name = self.config['database']['schema_name']
        table_name = self.config['database']['table_name']
        
        logger.info(f"Bước 1: Trích xuất dữ liệu từ bảng PostgreSQL: {schema_name}.{table_name}")
        
        # Lấy connection từ lớp cha BasePostgresExtractor (đã tích hợp sẵn src.postgres.core.connection)
        conn = self._get_connection()
        query = f"SELECT * FROM {schema_name}.{table_name}"
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Ép kiểu và làm sạch cơ bản bằng phương thức có sẵn của base_extractor (tuỳ chọn)
        df = self._convert_data_types(df)
        
        return df

    def process_and_split_data(self, df: pd.DataFrame):
        logger.info("Bước 2: Xử lý feature và loại bỏ cột không cần thiết (Drop Columns)")
        cols_to_drop = self._read_schema()
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

        logger.info("Bước 3: Chia tách dữ liệu Train/Test/Validation")
        train_ratio = self.config['data_split']['train_ratio']
        test_ratio = self.config['data_split']['test_ratio']
        val_ratio = self.config['data_split']['validation_ratio']
        
        # Cân bằng tỷ lệ chia
        test_size_relative = test_ratio / (test_ratio + val_ratio)
        
        train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=42)
        val_df, test_df = train_test_split(temp_df, train_size=test_size_relative, random_state=42)
        
        return df, train_df, val_df, test_df

    def export_to_feature_store(self, full_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        feature_path = self.config['feature_store']['file_path']
        logger.info(f"Bước 4: Xuất dữ liệu ra Artifacts ({feature_path})")
        
        base_dir = os.path.dirname(feature_path)
        os.makedirs(base_dir, exist_ok=True)
        
        # Lưu file master features
        full_df.to_csv(feature_path, index=False)
        
        # Lưu split
        train_df.to_csv(os.path.join(base_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(base_dir, 'validation.csv'), index=False)
        test_df.to_csv(os.path.join(base_dir, 'test.csv'), index=False)
        
        if self.config['feature_store'].get('dvc_tracked', True):
            logger.info("Đang track dữ liệu qua DVC...")
            # Gọi DVC Handler add track file vào hệ thống
            # self.dvc.add_data(base_dir) # Hoặc phương thức track dữ liệu cụ thể từ DVCHandler của bạn

    def run(self):
        logger.info("Khởi động Calories Data Ingestion Pipeline")
        df = self.extract_data_from_postgresql()
        full_df, train_df, val_df, test_df = self.process_and_split_data(df)
        self.export_to_feature_store(full_df, train_df, val_df, test_df)
        logger.info("Quá trình Ingestion hoàn tất.")

if __name__ == "__main__":
    yaml_path = "config/pipelines/ml_calories/data_ingestion.yaml"
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
        
    pipeline = CaloriesDataIngestion(config)
    pipeline.run()