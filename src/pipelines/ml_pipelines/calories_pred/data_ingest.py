import os
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

        # THAY ĐỔI 1: Đổi tên biến cho đúng ngữ nghĩa
        self.dataset_schema = self._load_schema()

    def _load_schema(self) -> dict:
        schema_file = self.config['schema']['data_config_file']
        logger.info(f"Đọc cấu hình Dataset Schema từ YAML: {schema_file}")
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                # THAY ĐỔI 2: Đọc bằng yaml thay vì json
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Không tìm thấy file {schema_file}.")
            raise
        except yaml.YAMLError as exc:
            logger.error(f"Lỗi định dạng YAML trong file {schema_file}: {exc}")
            raise

    @monitor_performance()
    def extract_data_from_postgresql(self) -> pd.DataFrame:
        schema_name = self.config['database'].get('schema_name', 'raw')
        
        # THAY ĐỔI 3: Lấy table_name, fallback về schema yaml thay vì json
        table_name = self.config['database'].get('table_name') or self.dataset_schema.get('table_name')

        if not table_name:
            raise ValueError("Thiếu cấu hình tên bảng (table_name).")

        logger.info(f"Bước 1: Trích xuất dữ liệu từ bảng PostgreSQL: {schema_name}.{table_name}")
        
        query = f"SELECT * FROM {schema_name}.{table_name}"
        
        # THAY ĐỔI 4: Gọi trực tiếp fetch_as_dataframe từ Base Class để tận dụng:
        # - Ép kiểu array/decimal tự động
        # - Ghi nhận Dataset Lineage lên MLflow
        # - Đặt index trực tiếp
        df = self.fetch_as_dataframe(
            query=query,
            index_col='recipe_id', # Ánh xạ cột recipe_id thành Index
            mlflow_tracker=self.tracker,
            dataset_name=f"{schema_name}_{table_name}"
        )
            
        return df

    @monitor_performance()
    def process_and_split_data(self, df: pd.DataFrame):
        logger.info("Bước 2 & 3: Lọc features và chia tách dữ liệu Train/Test/Validation")
        
        cols_to_drop = self.config['schema'].get('columns_to_drop', [])
        
        if cols_to_drop:
            logger.info(f"Tiến hành loại bỏ các cột: {cols_to_drop}")
            df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

        train_ratio = self.config['data_split']['train_ratio']
        test_ratio = self.config['data_split']['test_ratio']
        val_ratio = self.config['data_split']['validation_ratio']
        
        test_size_relative = test_ratio / (test_ratio + val_ratio)
        
        train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=42)
        val_df, test_df = train_test_split(temp_df, train_size=test_size_relative, random_state=42)
        
        return df, train_df, val_df, test_df

    @monitor_performance()
    def export_to_feature_store(self, df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        # THAY ĐỔI 5: Fix lỗi logic mặc định 'True' thành đường dẫn chuỗi (string)
        export_dir = self.config['feature_store'].get('export_dir', "Artifacts/features/calories")
        dvc_tracked = self.config['feature_store'].get('dvc_tracked', True)

        os.makedirs(export_dir, exist_ok=True)
        
        files = {
            "current_data.csv": df,
            "train.csv": train_df,
            "val.csv": val_df,
            "test.csv": test_df
        }

        for file_name, dataset in files.items():
            file_path = os.path.join(export_dir, file_name)
            dataset.to_csv(file_path, index=True) # Lưu ý: Để index=True nếu recipe_id đang là Index
            logger.info(f"Đã lưu: {file_path}")
            
            if dvc_tracked:
                self.dvc.add_data(file_path)

        logger.info("Hoàn tất xuất dữ liệu ra Feature Store.")

    def extract(self) -> pd.DataFrame:
        """Implement the abstract method required by BasePostgresExtractor."""
        return self.extract_data_from_postgresql()

    def run(self):
        logger.info("Khởi động Calories Data Ingestion Pipeline")
        
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