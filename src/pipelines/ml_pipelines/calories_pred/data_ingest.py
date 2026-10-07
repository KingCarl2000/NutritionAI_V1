import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Import module từ kiến trúc dự án
from src.postgres.core.connection import get_connection
from src.nutrition_core.logging.logger import logger
from src.data_io.dvc_handler import DVCHandler # Giả sử bạn có class này
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker
from src.nutrition_core.logging.logger import monitor_performance
from src.nutrition_core.exception.exception import DataTransformationError

@monitor_performance
class CaloriesDataIngestion:
    def __init__(self, output_dir: str = "Artifacts/data/calories"):
        """
        Khởi tạo pipeline với thư mục đầu ra để lưu các file CSV.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.table_name = "calories"

    def read_data_from_postgres(self) -> pd.DataFrame:
        """Bước 1 & 2: PostgreSQL => Read data : calories => Convert to DataFrame"""
        logger.info(f"Bắt đầu trích xuất dữ liệu từ bảng '{self.table_name}' trong PostgreSQL")
        try:
            conn = get_connection()
            # Trích xuất toàn bộ dữ liệu từ bảng calories
            query = f"SELECT * FROM {self.table_name};"
            df = pd.read_sql(query, conn)
            conn.close()
            
            logger.info(f"Đọc thành công {len(df)} bản ghi từ cơ sở dữ liệu.")
            return df
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất dữ liệu từ PostgreSQL: {str(e)}")
            raise e

    def save_raw_data(self, df: pd.DataFrame):
        """Bước 3: Save raw.csv"""
        raw_path = os.path.join(self.output_dir, "raw.csv")
        df.to_csv(raw_path, index=False)
        logger.info(f"Đã lưu dữ liệu thô vào: {raw_path}")

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bước 4: Feature Engineering (basic preprocessing)"""
        logger.info("Thực hiện Feature Engineering cơ bản...")
        
        # 4.1 Xóa các bản ghi có giá trị null
        df = df.dropna()
        
        # 4.2 Loại bỏ các cột không có giá trị dự đoán (ví dụ: User_ID)
        if 'User_ID' in df.columns:
            df = df.drop(columns=['User_ID'])
            
        # 4.3 Mã hóa (Encode) biến phân loại Gender thành số
        if 'Gender' in df.columns:
            # Chuẩn hóa chữ thường/hoa và map sang 0/1
            df['Gender'] = df['Gender'].str.lower().map({'male': 0, 'female': 1})
            # Xóa các dòng có Gender không hợp lệ (NaN sau khi map)
            df = df.dropna(subset=['Gender'])
            
        logger.info(f"Kích thước dữ liệu sau khi tiền xử lý: {df.shape}")
        return df

    def split_and_save_data(self, df: pd.DataFrame):
        """Bước 5 & 6: Train-Test Split => (Save train.csv, Save test.csv)"""
        logger.info("Thực hiện phân chia tập dữ liệu (Train: 80%, Test: 20%)...")
        
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        train_path = os.path.join(self.output_dir, "train.csv")
        test_path = os.path.join(self.output_dir, "test.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Đã lưu tập Train ({len(train_df)} bản ghi) tại: {train_path}")
        logger.info(f"Đã lưu tập Test ({len(test_df)} bản ghi) tại: {test_path}")

    def run_pipeline(self):
        """Hàm thực thi toàn bộ quy trình pipeline"""
        logger.info("========== BẮT ĐẦU PIPELINE CALORIES DATA INGESTION ==========")
        try:
            # 1. PostgreSQL => Read data : calories => Convert to DataFrame
            raw_df = self.read_data_from_postgres()
            
            # 2. Save raw.csv
            self.save_raw_data(raw_df)
            
            # 3. Feature Engineering (basic preprocessing)
            processed_df = self.feature_engineering(raw_df)
            
            # 4. Train-Test Split => (Save train.csv, Save test.csv)
            self.split_and_save_data(processed_df)
            
            logger.info("========== HOÀN THÀNH PIPELINE CALORIES DATA INGESTION ==========")
        except Exception as e:
            logger.error(f"Pipeline thất bại với lỗi: {str(e)}")

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Đọc danh sách cột cần xoá từ config thay vì hardcode
        cols_to_drop = self.config.get("columns_to_drop", [])
        df = df.drop(columns=cols_to_drop, errors='ignore')
        # ... các bước xử lý khác ...
        return df
    except Exception as e:
        raise DataTransformationError(f"Lỗi khi transform dữ liệu: {e}")

    def run_pipeline(self):
        with MLOpsTracker(experiment_name="Calories_Ingestion").start_run():
            # ... chạy các bước ...
            # Log metadata
            mlflow.log_param("test_split_ratio", 0.2)
            mlflow.log_metric("train_samples", len(train_df))
        
             # DVC Tracking
            dvc = DVCHandler()
            dvc.add_and_push([train_path, test_path])

if __name__ == "__main__":
    pipeline = CaloriesDataIngestion()
    pipeline.run_pipeline()