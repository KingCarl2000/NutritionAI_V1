import os
import sys
import yaml
import numpy as np
import io
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from psycopg.errors import UndefinedTable

# Import connections and core
from src.postgres.core.connection import get_connection
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.exception.exception import (
    DataPipelineException,
    DataTransformationError,
    DataCleansingError
)

# =====================================================================
# [QUAN TRỌNG] ĐẶT CÁC HÀM NÀY Ở NGOÀI CLASS ĐỂ HỖ TRỢ MULTIPROCESSING
# =====================================================================
def q1(x): return x.quantile(0.25)
def q3(x): return x.quantile(0.75)
def get_mode(x): 
    m = x.mode()
    return m.iloc[0] if not m.empty else np.nan
def get_skew(x): return x.skew()
def get_kurt(x): return x.kurt()

def process_agg_chunk(args):
    """
    Hàm Worker: Thực thi tính toán thống kê trên từng mảnh (chunk) dữ liệu.
    """
    df_chunk, num_cols, table_name = args
    
    # 1. Giữ nguyên 11 hàm thống kê ban đầu
    agg_dict = {
        col: ['mean', 'max', 'min', 'std', 'median', 'var', get_mode, q1, q3, get_skew, get_kurt]
        for col in num_cols
    }
    
    grouped = df_chunk.groupby(['Id', 'date']).agg(agg_dict).reset_index()
    
    # 2. Đổi tên cột để tránh nhầm lẫn: TênCột_HàmThốngKê_TênBảng
    new_cols = []
    for col in grouped.columns:
        if col[0] in ['Id', 'date']:
            new_cols.append(col[0])  # Giữ nguyên Id và date để join
        else:
            # col[0]: Tên cột gốc, col[1]: Hàm thống kê
            new_cols.append(f"{col[0]}_{col[1]}_{table_name}")
            
    grouped.columns = new_cols
    return grouped

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
        
        # Mở rộng danh sách nhận diện các biến thể tên cột thời gian
        possible_date_keywords = ['activitydate', 'date', 'activityhour', 'activityminute', 'time', 'activity_date']
        
        for col in cat_cols:
            if col.lower() in possible_date_keywords:
                return col
                
        # Fallback kiểm tra trực tiếp trên các cột của DataFrame nếu schema không khớp
        if table_name in self.dataframes:
            for col in self.dataframes[table_name].columns:
                if col.lower() in possible_date_keywords:
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
    # PHASE 1: INGESTION & CHUẨN HÓA CƠ SỞ (OPTIMIZED TIMESTAMP)
    # ==========================================
    def phase_1_ingestion_and_standardize(self):
        logger.info("Phase 1: Chuẩn hóa trục thời gian và Load Data (Optimized via SQL Timestamp)...")
        with get_connection() as conn:
            for table_name in self.tables_config.keys():
                try:
                    date_col = self._get_date_column_name(table_name)
                    
                    if date_col:
                        # Ép kiểu tường minh cột thời gian thành TIMESTAMP trực tiếp trong SQL
                        # Giúp COPY truyền sang client dạng giá trị chuẩn, giảm hoàn toàn chi phí parse của pandas
                        query = f"""
                            COPY (
                                SELECT *, CAST("{date_col}" AS TIMESTAMP) AS date_parsed 
                                FROM raw."{table_name}"
                            ) TO STDOUT WITH CSV HEADER
                        """
                    else:
                        query = f'COPY raw."{table_name}" TO STDOUT WITH CSV HEADER'
                    
                    # Khởi tạo bộ đệm nhị phân trên RAM
                    buffer = io.BytesIO()
                    
                    with conn.cursor() as cur:
                        with cur.copy(query) as copy:
                            for data in copy:
                                buffer.write(data)
                                
                    buffer.seek(0)
                    
                    # Đọc dữ liệu thô vào Pandas
                    df = pd.read_csv(buffer, low_memory=False)
                    
                    # Nếu có cột date_parsed vừa tạo từ SQL, gán lại thành cột 'date' chính thức
                    if date_col and 'date_parsed' in df.columns:
                        if date_col in df.columns:
                            df = df.drop(columns=[date_col])
                        df.rename(columns={'date_parsed': 'date'}, inplace=True)
                        
                        # Chuyển đổi siêu tốc sang datetime64[ns] và chuẩn hóa về 00:00:00
                        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
                        
                    self.dataframes[table_name] = df
                    logger.info(f"Đã chuẩn hóa bảng: {table_name} (Shape: {df.shape})")
                    
                except Exception as e:
                    logger.warning(f"Không thể đọc bảng {table_name} từ schema raw: {e}")
                    raise DataTransformationError(f"Lỗi trích xuất bảng {table_name}: {e}", sys) 

    # ==========================================
    # PHASE 2: FEATURE ENGINEERING (PARALLEL)
    # ==========================================
    def phase_2_feature_engineering(self):
        try:
            logger.info("Phase 2: Bắt đầu Aggregation với xử lý song song (Chunking)...")
            aggregated_dfs = {}
            weight_df = None
            
            # Khởi tạo số lượng worker dựa trên số nhân CPU của máy (bớt 1 nhân cho hệ điều hành)
            n_workers = max(1, multiprocessing.cpu_count() - 1)
            logger.info(f"Sử dụng {n_workers} CPU cores để tính toán song song.")
            
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
                            # Tách dữ liệu thành các chunks an toàn dựa trên tập hợp Id duy nhất
                            # (Đảm bảo dữ liệu của 1 user không bị cắt làm đôi giữa 2 CPU core)
                            unique_ids = df['Id'].unique()
                            id_chunks = np.array_split(unique_ids, n_workers)
                            
                            chunks = []
                            for id_chunk in id_chunks:
                                df_chunk = df[df['Id'].isin(id_chunk)]
                                chunks.append((df_chunk, num_cols, table_name))
                                
                            # Chạy đa luồng bằng ProcessPoolExecutor
                            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                                results = list(executor.map(process_agg_chunk, chunks))
                                
                            # Gộp kết quả từ các core lại thành 1 DataFrame hoàn chỉnh
                            grouped = pd.concat(results, ignore_index=True)
                            
                            aggregated_dfs[table_name] = grouped
                            logger.info(f"Đã Aggregate xong bảng: {table_name} (Shape: {grouped.shape})")
                            
                            # --------------------------------------------------
                            # LƯU DỮ LIỆU TRUNG GIAN (ARTIFACTS CỤC BỘ)
                            # --------------------------------------------------
                            inter_dir = r"D:\NutritionAI_V1\Data\intermediate"
                            os.makedirs(inter_dir, exist_ok=True)
                            
                            # Lưu Parquet local để kiểm tra
                            grouped.to_parquet(os.path.join(inter_dir, f"{table_name}_agg.parquet"), index=False)
                            
                            # --------------------------------------------------
                            # KHUNG LỆNH ĐẨY LÊN DATABASE (POSTGRESQL STAGING)
                            # --------------------------------------------------
                            # with get_connection() as conn:
                            #     grouped.to_sql(name=f"{table_name}_aggregated", con=conn, schema="staging", if_exists="replace", index=False)
                            # logger.info(f"# [DB] Đã push bảng staging.{table_name}_aggregated lên PostgreSQL")
                            
            # Nội suy dữ liệu Cân nặng (Weight Interpolation)
            if weight_df is not None and 'dailyActivity_merged' in self.dataframes:
                logger.info("Đang xử lý nội suy bảng weightLogInfo_merged...")
                daily_df = self.dataframes['dailyActivity_merged']
                time_grid = daily_df[['Id', 'date']].drop_duplicates()
                
                weight_df = weight_df.groupby(['Id', 'date']).mean(numeric_only=True).reset_index()
                weight_interpolated = pd.merge(time_grid, weight_df, on=['Id', 'date'], how='left')
                weight_interpolated = weight_interpolated.sort_values(by=['Id', 'date'])
                
                weight_cols = [col for col in weight_interpolated.select_dtypes(include=[np.number]).columns if col != 'Id']
                
                for col in weight_cols:
                    weight_interpolated[col] = weight_interpolated.groupby('Id')[col].transform(
                        lambda x: x.interpolate(method='linear').ffill().bfill()
                    )
                
                # Sửa lại tên các cột weight để đồng nhất format (ngoại trừ Id, date)
                weight_interpolated.rename(columns={col: f"{col}_interpolated_weightLogInfo" for col in weight_cols}, inplace=True)
                weight_df = weight_interpolated
                
                # Lưu file cân nặng đã nội suy ra Local
                inter_dir = r"D:\NutritionAI_V1\Data\intermediate"
                os.makedirs(inter_dir, exist_ok=True)
                weight_df.to_parquet(os.path.join(inter_dir, "weight_interpolated.parquet"), index=False)
                
                # # [DB Push cho bảng Weight]
                # # with get_connection() as conn:
                # #     weight_df.to_sql(name="weight_interpolated", con=conn, schema="staging", if_exists="replace", index=False)

            return aggregated_dfs, weight_df

        except Exception as e:
            raise DataTransformationError(f"Lỗi quá trình Feature Engineering (Phase 2 Parallel): {e}", sys)

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

            # Tạo thư mục chứa báo cáo đánh giá
            report_dir = r"D:\NutritionAI_V1\Data\reports"
            os.makedirs(report_dir, exist_ok=True)
            report_file = os.path.join(report_dir, "data_assessment_report.txt")
            
            with open(report_file, "w", encoding="utf-8") as f:
                f.write("========== BÁO CÁO ĐÁNH GIÁ MASTER DATAFRAME ==========\n\n")
                f.write(f"1. Kích thước (Shape): {master_df.shape[0]} dòng, {master_df.shape[1]} cột\n\n")
                
                f.write("2. Tỉ lệ Dữ liệu khuyết (Missing %):\n")
                f.write(missing_cols.round(2).to_string() if not missing_cols.empty else "Không có missing value.")
                f.write("\n\n")
                
                f.write("3. Tỉ lệ Giá trị 0 (%):\n")
                f.write(zero_cols.round(2).to_string() if not zero_cols.empty else "Không có giá trị 0.")
                f.write("\n\n=======================================================")
            
                logger.info(f"Đã xuất báo cáo chi tiết ra file: {report_file}")
            return master_df

        except Exception as e:
            # Vẫn dùng DataCleansingError để track đúng phase
            raise DataCleansingError(f"Lỗi Integration và Khảo sát: {e}", sys)

    # ==========================================
    # PHASE 4: TỐI ƯU HÓA & ĐÓNG GÓI
    # ==========================================
    def phase_4_optimization_and_packaging(self, df):
        try:
            logger.info("Phase 4: Lọc đa cộng tuyến thông minh (So sánh với Target)...")
            target_col = 'Calories'
            
            numeric_df = df.select_dtypes(include=[np.number])
            
            # 1. Tính ma trận tương quan toàn cục
            corr_matrix = numeric_df.corr().abs()
            
            # 2. Lấy riêng mức độ tương quan của từng biến đối với cột Target ('Calories')
            target_corr = corr_matrix[target_col]
            
            # 3. Tạo ma trận tương quan chỉ giữa các features (loại bỏ Id và Target để xét chéo nhau)
            features_corr = corr_matrix.drop(index=[target_col, 'Id'], columns=[target_col, 'Id'], errors='ignore')
            upper = features_corr.where(np.triu(np.ones(features_corr.shape), k=1).astype(bool))
            
            to_drop = set()
            
            # 4. Duyệt ma trận để tìm các cặp đa cộng tuyến (> 0.85)
            for col in upper.columns:
                for row in upper.index:
                    if upper.loc[row, col] > 0.85:
                        # [SMART DROP] Biến nào tương quan với Calories YẾU HƠN sẽ bị xóa
                        if target_corr.get(col, 0) < target_corr.get(row, 0):
                            to_drop.add(col)
                        else:
                            to_drop.add(row)
                            
            to_drop = list(to_drop)
            df_final = df.drop(columns=to_drop)
            
            logger.info(f"Đã so sánh với Target và xóa {len(to_drop)} cột bị đa cộng tuyến (>0.85).")
            
            # 5. Đóng gói ra thư mục Artifacts
            output_dir = r"D:\NutritionAI_V1\Data\processed"
            os.makedirs(output_dir, exist_ok=True)
            
            # Lưu Parquet (Khuyên dùng cho Model Training vì nhẹ và load nhanh)
            parquet_path = os.path.join(output_dir, "core_daily_calories_features.parquet")
            df_final.to_parquet(parquet_path, index=False)
            
            # Lưu CSV (Để dễ dàng xem bằng Excel/VSCode)
            csv_path = os.path.join(output_dir, "core_daily_calories_features.csv")
            df_final.to_csv(csv_path, index=False)
            
            logger.info(f"Đã xuất file Final Dataset (Parquet & CSV) sẵn sàng cho Model tại: {output_dir}")
            
            # --------------------------------------------------
            # KHUNG LỆNH ĐẨY LÊN DATABASE (POSTGRESQL CORE)
            # --------------------------------------------------
            # with get_connection() as conn:
            #     df_final.to_sql(name="daily_calories_features", con=conn, schema="core", if_exists="replace", index=False)
            # logger.info("# [DB] Đã push bảng core.daily_calories_features lên PostgreSQL")
            
            return df_final
            
        except Exception as e:
            raise DataTransformationError(f"Lỗi tối ưu hóa đa cộng tuyến và Đóng gói: {e}", sys)

if __name__ == "__main__":
    pipeline = DataTransformationPipeline()
    final_dataset = pipeline.execute_pipeline()
    print(final_dataset.head())



