import os
import sys
import yaml
import pandas as pd
import numpy as np
from psycopg.errors import UndefinedTable

# Import connection pool từ architecture hiện tại
from src.postgres.core.connection import get_connection

# 1. Tích hợp Custom Logger và Exception từ nutrition_core
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.exception.exception import (
    DataPipelineException,
    DataTransformationError,
    DataCleansingError
)

class DataTransformationPipeline:
    def __init__(self, schema_path: str = r'D:\NutritionAI_V1\src\api\data_schema\schema.yaml'):
        self.schema_path = schema_path
        self.schema_config = self._load_schema()
        self.tables_config = self.schema_config.get('Fitabase Data 3.12.16-4.11.16', {})
        self.dataframes = {}

    def _load_schema(self):
        try:
            with open(self.schema_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise DataPipelineException(f"Lỗi khi đọc file schema: {e}", sys)

    def _get_date_column_name(self, table_name):
        config = self.tables_config.get(table_name, {})
        cat_cols = config.get('categorical_columns', [])
        for col in cat_cols:
            if col.lower() in ["ActivityDate", "ActivityHour", "ActivityMinute", "Date", "Time", "date"]:
                return col
        return None

    def execute_pipeline(self):
        try:
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
        except Exception as e:
            logger.error(f"Pipeline thất bại: {e}")
            raise DataPipelineException(str(e), sys)

    # ==========================================
    # PHASE 1: INGESTION & CHUẨN HÓA CƠ SỞ
    # ==========================================
    def phase_1_ingestion_and_standardize(self):
        logger.info("Phase 1: Chuẩn hóa trục thời gian và Load Data...")
        with get_connection() as conn:
            for table_name in self.tables_config.keys():
                try:
                    # SỬA DÒNG NÀY: Thêm dấu ngoặc kép (") bao quanh table_name
                    query = f'SELECT * FROM raw."{table_name}"'
                    
                    df = pd.read_sql(query, conn)
                    
                    date_col = self._get_date_column_name(table_name)
                    if date_col and date_col in df.columns:
                        df['date_temp'] = pd.to_datetime(df[date_col], format='mixed', errors='coerce')
                        df = df.drop(columns=[date_col])
                        df.rename(columns={'date_temp': 'date'}, inplace=True)
                        df['date'] = df['date'].dt.date
                        
                    self.dataframes[table_name] = df
                    logger.info(f"Đã chuẩn hóa bảng: {table_name}")
                except Exception as e:
                    logger.warning(f"Không thể đọc bảng {table_name} từ schema raw: {e}")
                    raise DataTransformationError(f"Lỗi trích xuất bảng {table_name}: {e}", sys)
    
    # ==========================================
    # PHASE 2: FEATURE ENGINEERING
    # ==========================================
    def phase_2_feature_engineering(self):
        try:
            logger.info("Phase 2: Aggregation và Nội suy dữ liệu...")
            aggregated_dfs = {}
            weight_df = None
            
            for table_name, df in self.dataframes.items():
                if table_name == 'dailyActivity_merged':
                    continue
                elif table_name == 'weightLogInfo_merged':
                    weight_df = df
                else:
                    if 'date' in df.columns and 'Id' in df.columns:
                        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        num_cols = [col for col in num_cols if col != 'Id']
                        
                        if num_cols:
                            # 1. Các custom function
                            def q1(x): return x.quantile(0.25)
                            def q3(x): return x.quantile(0.75)
                            
                            # Hàm xử lý mode
                            def get_mode(x): 
                                m = x.mode()
                                return m.iloc[0] if not m.empty else np.nan
                                
                            # Hàm xử lý độ lệch (skewness) và độ nhọn (kurtosis)
                            def get_skew(x): return x.skew()
                            def get_kurt(x): return x.kurt()
                            
                            # 2. Đưa các hàm này vào agg_dict thay vì dùng string
                            agg_dict = {col: [
                                'mean', 'max', 'min', 'std', 'median', 'var', 
                                get_mode, q1, q3, get_skew, get_kurt
                            ] for col in num_cols}
                            
                            grouped = df.groupby(['Id', 'date']).agg(agg_dict).reset_index()
                            
                            # 3. Làm phẳng (flatten) các cột MultiIndex
                            grouped.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped.columns]
                            
                            aggregated_dfs[table_name] = grouped
                            logger.info(f"Đã Aggregate bảng: {table_name}")

                            
            if weight_df is not None and 'dailyActivity_merged' in self.dataframes:
                daily_df = self.dataframes['dailyActivity_merged']
                time_grid = daily_df[['Id', 'date']].drop_duplicates()
                
                weight_df = weight_df.groupby(['Id', 'date']).mean(numeric_only=True).reset_index()
                weight_interpolated = pd.merge(time_grid, weight_df, on=['Id', 'date'], how='left')
                weight_interpolated = weight_interpolated.sort_values(by=['Id', 'date'])
                
                weight_cols = weight_interpolated.select_dtypes(include=[np.number]).columns.tolist()
                weight_cols.remove('Id')
                
                weight_interpolated['date'] = pd.to_datetime(weight_interpolated['date'])
                weight_interpolated.set_index('date', inplace=True)
                
                for col in weight_cols:
                    weight_interpolated[col] = weight_interpolated.groupby('Id')[col].transform(
                        lambda x: x.interpolate(method='nearest').ffill().bfill()
                    )
                weight_interpolated.reset_index(inplace=True)
                
            return aggregated_dfs, weight_df
        except Exception as e:
            raise DataTransformationError(f"Lỗi quá trình Feature Engineering: {e}", sys)

    # ==========================================
    # PHASE 3: INTEGRATION & LÀM SẠCH (Chỉ Thống kê Đánh giá)
    # ==========================================
    def phase_3_integration_and_cleaning(self, aggregated_dfs, weight_df):
        try:
            logger.info("Phase 3: Master Join và Khảo sát dữ liệu...")
            master_df = self.dataframes['dailyActivity_merged'].copy()
            
            # Merge các bảng Aggregate
            for table_name, agg_df in aggregated_dfs.items():
                master_df = pd.merge(master_df, agg_df, on=['Id', 'date'], how='left', suffixes=('', f'_{table_name}'))
                
            # Merge bảng Cân nặng
            if weight_df is not None:
                master_df = pd.merge(master_df, weight_df, on=['Id', 'date'], how='left')
                
            # ĐÁNH GIÁ DATAFRAME MỚI
            logger.info("---------- BÁO CÁO ĐÁNH GIÁ MASTER DATAFRAME ----------")
            
            # 1. Kích thước tập dữ liệu
            logger.info(f"[1] Kích thước (Shape): {master_df.shape[0]} dòng, {master_df.shape[1]} cột")
            
            # 2. Tính tỉ lệ phần trăm Missing Values
            missing_pct = (master_df.isnull().sum() / len(master_df)) * 100
            missing_cols = missing_pct[missing_pct > 0].sort_values(ascending=False)
            if not missing_cols.empty:
                logger.info(f"[2] Tỉ lệ Dữ liệu khuyết (Missing %):\n{missing_cols.round(2).to_string()}")
            else:
                logger.info("[2] Tỉ lệ Dữ liệu khuyết: Không có.")
                
            # 3. Tính tỉ lệ phần trăm Giá trị 0
            zero_pct = (master_df == 0).sum() / len(master_df) * 100
            zero_cols = zero_pct[zero_pct > 0].sort_values(ascending=False)
            if not zero_cols.empty:
                logger.info(f"[3] Tỉ lệ Giá trị 0 (%):\n{zero_cols.round(2).to_string()}")
            else:
                logger.info("[3] Tỉ lệ Giá trị 0: Không có.")
                
            logger.info("-------------------------------------------------------")
            logger.info("Tạm dừng các bước làm sạch (Imputation/Drop). Giữ nguyên trạng thái thô sau Merge.")
            
            return master_df
        
        except Exception as e:
            # Vẫn dùng DataCleansingError để track đúng phase
            raise DataCleansingError(f"Lỗi Integration và Khảo sát: {e}", sys)

    # ==========================================
    # PHASE 4: TỐI ƯU HÓA & ĐÓNG GÓI
    # ==========================================
    def phase_4_optimization_and_packaging(self, df):
        try:
            logger.info("Phase 4: Lọc đa cộng tuyến và Đóng gói...")
            target_col = 'Calories'
            
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
            to_drop = [col for col in to_drop if col not in [target_col, 'Id']]
            
            df_final = df.drop(columns=to_drop)
            logger.info(f"Đã xóa các cột bị đa cộng tuyến (>0.85): {to_drop}")
            
            output_dir = "D:\\NutritionAI_V1\\Data\\processed"
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, "core_daily_calories_features.csv")
            df_final.to_csv(csv_path, index=False)
            
            logger.info(f"Đã xuất file CSV sẵn sàng cho Model tại: {csv_path}")
            
            # self._save_to_core_schema(df_final, "daily_calories_features")
            return df_final
        except Exception as e:
            # Bắt DataTransformationError cho tiến trình drop columns/lọc đa cộng tuyến
            raise DataTransformationError(f"Lỗi tối ưu hóa đa cộng tuyến và Đóng gói: {e}", sys)

if __name__ == "__main__":
    pipeline = DataTransformationPipeline()
    final_dataset = pipeline.execute_pipeline()
    print(final_dataset.head())

