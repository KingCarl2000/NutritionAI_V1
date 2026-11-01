import os
import json
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

from src.nutrition_core.logging.logger import logger, monitor_performance
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker
from src.data_io.base_extractor import BasePostgresExtractor
from src.data_io.dvc_handler import DVCHandler

class CaloriesDataIngestion(BasePostgresExtractor):
    def __init__(self, config: dict, tracker: MLOpsTracker = None):
        super().__init__()
        self.config = config
        self.dvc = DVCHandler()
        
        # Nhận tracker từ Orchestrator để đồng bộ run_id, nếu không có mới tự tạo
        self.tracker = tracker if tracker else MLOpsTracker(experiment_name="calories_pred")

        self.json_schema = self._load_json_schema()

    def _load_json_schema(self) -> dict:
        schema_file = self.config['schema']['json_schema_file']
        logger.info(f"Đọc cấu hình JSON Schema từ: {schema_file}")
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Không tìm thấy file {schema_file}.")
            raise
        except json.JSONDecodeError:
            logger.error(f"Lỗi định dạng JSON trong file {schema_file}.")
            raise

    @monitor_performance()
    def extract_data_from_postgresql(self) -> pd.DataFrame:
        # THAY ĐỔI: Đọc schema và table_name từ file cấu hình YAML thay vì JSON
        schema_name = self.config['database'].get('schema_name', 'public')
        table_name = self.config['database'].get('table_name')
        
        if not table_name:
            # Lấy fallback từ json nếu yaml không có
            table_name = self.json_schema.get('source_dataset')

        if not table_name:
            raise ValueError("Thiếu cấu hình tên bảng (table_name hoặc source_dataset).")

        logger.info(f"Bước 1: Trích xuất dữ liệu từ bảng PostgreSQL: {schema_name}.{table_name}")
        
        conn = self._get_connection()
        query = f"SELECT * FROM {schema_name}.{table_name}"
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Nếu class base có hàm _convert_data_types, gọi nó ở đây
        if hasattr(self, '_convert_data_types'):
            df = self._convert_data_types(df)
            
        return df

    @monitor_performance()
    def process_and_split_data(self, df: pd.DataFrame):
        logger.info("Bước 2 & 3: Lọc features và chia tách dữ liệu Train/Test/Validation")
        
        # MỚI: Lấy danh sách cột cần loại bỏ từ YAML Config thay vì JSON
        cols_to_drop = self.config['schema'].get('columns_to_drop', [])
        
        if cols_to_drop:
            logger.info(f"Tiến hành loại bỏ các cột: {cols_to_drop}")
            df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

        # ... (Phần chia tách train/test/val giữ nguyên vì đã lấy đúng từ self.config['data_split'])
        train_ratio = self.config['data_split']['train_ratio']
        test_ratio = self.config['data_split']['test_ratio']
        val_ratio = self.config['data_split']['validation_ratio']
        
        test_size_relative = test_ratio / (test_ratio + val_ratio)
        
        train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=42)
        val_df, test_df = train_test_split(temp_df, train_size=test_size_relative, random_state=42)
        
        return df, train_df, val_df, test_df

    @monitor_performance()
    def export_to_feature_store(self, df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        # MỚI: Lấy cấu hình xuất file từ YAML
        export_dir = self.config['feature_store'].get('export_dir', True)
        dvc_tracked = self.config['feature_store'].get('dvc_tracked', True)

        os.makedirs(export_dir, exist_ok=True)
        
        # Định nghĩa các đường dẫn
        files = {
            "current_data.csv": df,
            "train.csv": train_df,
            "val.csv": val_df,
            "test.csv": test_df
        }

        # Lưu file
        for file_name, dataset in files.items():
            file_path = os.path.join(export_dir, file_name)
            dataset.to_csv(file_path, index=False)
            logger.info(f"Đã lưu: {file_path}")
            
            # Khai báo DVC nếu được bật trong YAML
            if dvc_tracked:
                self.dvc.add_data(file_path)

        logger.info("Hoàn tất xuất dữ liệu ra Feature Store.")

    def extract(self) -> pd.DataFrame:
        """Implement the abstract method required by BasePostgresExtractor."""
        return self.extract_data_from_postgresql()

    def run(self):
        logger.info("Khởi động Calories Data Ingestion Pipeline")
        
        # Mở một run trên MLflow để tracking
        with self.tracker.start_run(run_name="data_ingestion"):
            df = self.extract()
            full_df, train_df, val_df, test_df = self.process_and_split_data(df)
            self.export_to_feature_store(full_df, train_df, val_df, test_df)
            
            self.tracker.log_params_and_metrics(
                params={
                    "train_ratio": self.config['data_split']['train_ratio'],
                    "test_ratio": self.config['data_split']['test_ratio'],
                    "validation_ratio": self.config['data_split']['validation_ratio'],
                    "dvc_tracked": self.config['feature_store'].get('dvc_tracked', True)
                },
                metrics={
                    "total_rows": len(full_df),
                    "total_features": len(full_df.columns),
                    "train_rows": len(train_df),
                    "validation_rows": len(val_df),
                    "test_rows": len(test_df)
                }
            )
        logger.info("Quá trình Ingestion hoàn tất và đã được ghi nhận trên MLflow.")

if __name__ == "__main__":
    yaml_path = "config/pipelines/ml_calories/data_ingestion.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    pipeline = CaloriesDataIngestion(config)
    pipeline.run()