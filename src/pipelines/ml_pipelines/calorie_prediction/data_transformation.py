import os
import yaml
import pandas as pd
import numpy as np
from psycopg.errors import UndefinedTable
import logging

# Import connection pool từ architecture hiện tại
from src.postgres.core.connection import get_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransformationPipeline:
    def __init__(self, schema_path: str = r'D:\NutritionAI_V1\src\api\data_schema\schema.yaml'):
        self.schema_path = schema_path
        self.schema_config = self._load_schema()
        self.tables_config = self.schema_config.get('Fitabase Data 3.12.16-4.11.16', {})
        self.dataframes = {}

    def _load_schema(self):
        """Đọc file schema.yaml để lấy cấu hình các bảng."""
        with open(self.schema_path, 'r') as file:
            return yaml.safe_load(file)

    def _get_date_column_name(self, table_name):
        """Xác định cột chứa ngày tháng dựa trên schema.yaml."""
        config = self.tables_config.get(table_name, {})
        cat_cols = config.get('categorical_columns', [])
        # Trong Fitabase, các cột thời gian thường nằm trong categorical_columns
        for col in cat_cols:
            if col.lower() in ['activitydate', 'date', 'activityhour', 'activityminute', 'time']:
                return col
        return None

    def execute_pipeline(self):
        """Điều phối thực thi 4 Phase của Pipeline."""
        logger.info("Bắt đầu Data Transformation Pipeline...")
        
        # Phase 1
        self.phase_1_ingestion_and_standardize()
        
        # Phase 2
        aggregated_dfs, weight_df = self.phase_2_feature_engineering()
        
        # Phase 3
        master_df = self.phase_3_integration_and_cleaning(aggregated_dfs, weight_df)
        
        # Phase 4
        final_df = self.phase_4_optimization_and_packaging(master_df)
        
        logger.info("Hoàn thành Data Transformation Pipeline!")
        return final_df

    # ==========================================
    # PHASE 1: INGESTION & CHUẨN HÓA CƠ SỞ
    # ==========================================
    def phase_1_ingestion_and_standardize(self):
        logger.info("Phase 1: Chuẩn hóa trục thời gian và Load Data...")
        
        with get_connection() as conn:
            for table_name in self.tables_config.keys():
                try:
                    # Đọc dữ liệu từ schema raw
                    query = f"SELECT * FROM raw.{table_name}"
                    df = pd.read_sql(query, conn)
                    
                    # Bước 2: Đồng nhất trục thời gian
                    date_col = self._get_date_column_name(table_name)
                    if date_col and date_col in df.columns:
                        # Convert sang datetime, sau đó lấy phần date (YYYY-MM-DD)
                        df['date_temp'] = pd.to_datetime(df[date_col], format='mixed', errors='coerce')
                        
                        # Xóa cột thời gian cũ và đổi tên cột mới thành 'date'
                        df = df.drop(columns=[date_col])
                        df.rename(columns={'date_temp': 'date'}, inplace=True)
                        
                        # Ép kiểu về datetime.date của Python (tương đương DATE của Postgres)
                        df['date'] = df['date'].dt.date
                    
                    self.dataframes[table_name] = df
                    logger.info(f" Đã chuẩn hóa bảng: {table_name}")
                    
                except Exception as e:
                    logger.warning(f"Không thể đọc bảng {table_name} từ schema raw: {e}")

    # ==========================================
    # PHASE 2: FEATURE ENGINEERING
    # ==========================================
    def phase_2_feature_engineering(self):
        logger.info("Phase 2: Aggregation và Nội suy dữ liệu...")
        aggregated_dfs = {}
        weight_df = None
        
        for table_name, df in self.dataframes.items():
            if table_name == 'dailyActivity_merged':
                continue # Bỏ qua bảng gốc
                
            elif table_name == 'weightLogInfo_merged':
                # Sẽ xử lý sau khi có khung thời gian từ dailyActivity
                weight_df = df
                
            else:
                # Bước 3: Aggregation cho dữ liệu Hourly/Minute
                if 'date' in df.columns and 'Id' in df.columns:
                    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    num_cols = [col for col in num_cols if col != 'Id'] # Không aggregate Id
                    
                    if num_cols:
                        # Khai báo các hàm thống kê
                        def q1(x): return x.quantile(0.25)
                        def q3(x): return x.quantile(0.75)
                        
                        agg_dict = {col: ['mean', 'max', 'min', 'sum', 'std', q1, q3] for col in num_cols}
                        
                        grouped = df.groupby(['Id', 'date']).agg(agg_dict).reset_index()
                        
                        # Làm phẳng Multi-index column (VD: Calories_sum)
                        grouped.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped.columns]
                        aggregated_dfs[table_name] = grouped
                        logger.info(f" Đã Aggregate bảng: {table_name}")

        # Bước 4: Nội suy dữ liệu Cân Nặng (Time-grid)
        if weight_df is not None and 'dailyActivity_merged' in self.dataframes:
            daily_df = self.dataframes['dailyActivity_merged']
            
            # Tạo Time-grid chuẩn từ dailyActivity
            time_grid = daily_df[['Id', 'date']].drop_duplicates()
            
            # Khử trùng lặp weightLog (nếu 1 ngày ghi cân nặng 2 lần, lấy trung bình)
            weight_df = weight_df.groupby(['Id', 'date']).mean(numeric_only=True).reset_index()
            
            # Merge weightLog vào Time-grid
            weight_interpolated = pd.merge(time_grid, weight_df, on=['Id', 'date'], how='left')
            
            # Nội suy: ffill() rồi bfill() theo từng Id
            weight_interpolated = weight_interpolated.sort_values(by=['Id', 'date'])
            weight_cols = weight_interpolated.select_dtypes(include=[np.number]).columns.tolist()
            weight_cols.remove('Id')
            
            # Đảm bảo cột date ở định dạng datetime để Pandas tính toán khoảng cách
            weight_interpolated['date'] = pd.to_datetime(weight_interpolated['date'])

            # Set date làm Index để thuật toán nearest hoạt động
            weight_interpolated.set_index('date', inplace=True)

            for col in weight_cols:
            # Dùng hàm interpolate với method='nearest'
            # Vẫn giữ ffill() và bfill() ở cuối để lo các khoảng trống ở tận cùng 2 đầu (nếu có)
                weight_interpolated[col] = weight_interpolated.groupby('Id')[col].transform(
                lambda x: x.interpolate(method='nearest').ffill().bfill()
            )

            # Reset lại index đưa date trở về thành 1 cột bình thường
            weight_interpolated.reset_index(inplace=True)
            
        return aggregated_dfs, weight_df

    # ==========================================
    # PHASE 3: INTEGRATION & LÀM SẠCH
    # ==========================================
    def phase_3_integration_and_cleaning(self, aggregated_dfs, weight_df):
        logger.info("Phase 3: Master Join và Kiểm toán dữ liệu khuyết...")
        
        # Bước 5: Master Join
        master_df = self.dataframes['dailyActivity_merged'].copy()
        
        # Merge các bảng Aggregate
        for table_name, agg_df in aggregated_dfs.items():
            master_df = pd.merge(master_df, agg_df, on=['Id', 'date'], how='left', suffixes=('', f'_{table_name}'))
            
        # Merge bảng Cân nặng
        if weight_df is not None:
            master_df = pd.merge(master_df, weight_df, on=['Id', 'date'], how='left')
            
        # Bước 6: Audit Missing Values
        # Tính % missing
        missing_pct = master_df.isnull().mean()
        cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
        
        master_df = master_df.drop(columns=cols_to_drop)
        logger.info(f" Đã xóa các cột thiếu >50%: {cols_to_drop}")
        
        # Điền các cột < 50% bằng Median của chính Id đó
        numeric_cols = master_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if master_df[col].isnull().sum() > 0:
                # Transform điền giá trị Median theo Id
                master_df[col] = master_df.groupby('Id')[col].transform(lambda x: x.fillna(x.median()))
                # Đề phòng trường hợp cả Id đó đều NaN, điền bằng Median toàn cục
                master_df[col] = master_df[col].fillna(master_df[col].median())
                
        logger.info(" Đã điền Missing Value bằng Median theo cá nhân (Id).")
        return master_df

    # ==========================================
    # PHASE 4: TỐI ƯU HÓA & ĐÓNG GÓI
    # ==========================================
    def phase_4_optimization_and_packaging(self, df):
        logger.info("Phase 4: Lọc đa cộng tuyến và Đóng gói...")
        
        # Cột mục tiêu cần giữ lại tuyệt đối
        target_col = 'Calories'
        
        # Bước 7: Lọc Đa Cộng Tuyến (Pearson > 0.85)
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()
        
        # Chọn tam giác trên của ma trận tương quan
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Tìm các cột có tương quan > 0.85
        to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
        
        # Đảm bảo không drop cột Target (Calories) hoặc Id
        to_drop = [col for col in to_drop if col not in [target_col, 'Id']]
        
        df_final = df.drop(columns=to_drop)
        logger.info(f" Đã xóa các cột bị đa cộng tuyến (>0.85): {to_drop}")
        
        # Bước 8: Lưu trữ Database & Xuất CSV
        output_dir = "Artifacts/data/processed"
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "core_daily_calories_features.csv")
        
        df_final.to_csv(csv_path, index=False)
        logger.info(f" Đã xuất file CSV sẵn sàng cho Model tại: {csv_path}")
        
        self._save_to_core_schema(df_final, "daily_calories_features")
        
        return df_final

    def _save_to_core_schema(self, df, table_name):
        """Lưu Dataframe vào schema 'core' trong PostgreSQL"""
        logger.info(f" Đang lưu dữ liệu vào schema core.{table_name}...")
        
        # Sử dụng SQLAlchemy engine (được khuyến nghị khi dùng Pandas to_sql)
        # Trong hệ thống của bạn, có thể cấu hình connection URI từ db_config
        from src.postgres.core.db_config import config
        from sqlalchemy import create_engine
        
        try:
            # Pandas yêu cầu sqlalchemy engine để thực thi to_sql mượt mà
            conn_string = config.get_connection_string().replace("postgresql://", "postgresql+psycopg://")
            engine = create_engine(conn_string)
            
            # Ghi đè nếu bảng đã tồn tại, lưu vào schema 'core'
            df.to_sql(table_name, engine, schema='core', if_exists='replace', index=False)
            logger.info(" Lưu Database thành công!")
        except Exception as e:
            logger.error(f" Lỗi khi lưu vào Database: {e}")

if __name__ == "__main__":
    pipeline = DataTransformationPipeline()
    final_dataset = pipeline.execute_pipeline()
    print(final_dataset.head())