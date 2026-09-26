import os
import sys
import yaml
import numpy as np
import io
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from psycopg.errors import UndefinedTable
from sqlalchemy import create_engine # Thêm sqlalchemy để pd.to_sql dễ dàng save to staging

# Import connections and core
from src.postgres.core.connection import get_connection, get_engine
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.exception.exception import (
    DataPipelineException, 
    DataTransformationError, 
    DataCleansingError
)

# =====================================================================
# CÁC HÀM THỐNG KÊ (HỖ TRỢ MULTIPROCESSING)
# =====================================================================
def q1(x):
    return x.quantile(0.25)
def q3(x):
    return x.quantile(0.75)
def get_mode(x):
    m = x.mode()
    return m.iloc[0] if not m.empty else np.nan
def get_skew(x):
    return x.skew()
def get_kurt(x):
    return x.kurt()

def process_agg_chunk(args):
    """
    Hàm Worker: Thực thi tính toán thống kê trên từng mảnh (chunk) dữ liệu 
    cho các bảng chi tiết hơn ngày (hourly, minute, heartrate...).
    """
    df_chunk, num_cols, table_name = args
    agg_dict = {col: ['mean', 'max', 'min', 'std', 'median', 'var', get_mode, q1, q3, get_skew, get_kurt] for col in num_cols}
    
    # Nhóm theo Id và date để đưa về cùng granularity với dailyActivity
    grouped = df_chunk.groupby(['Id', 'date']).agg(agg_dict).reset_index()
    
    new_cols = []
    for col in grouped.columns:
        if col[0] in ['Id', 'date']:
            new_cols.append(col[0])
        else:
            new_cols.append(f"{col[0]}_{col[1]}_{table_name}")
    grouped.columns = new_cols
    return grouped


class DataTransformationPipeline:
    def __init__(self, schema_path: str = None):
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        if schema_path is None:
            self.schema_path = str(self.project_root / 'docs' / 'architecture' / 'tables_created_report.yaml')
        else:
            self.schema_path = schema_path
            
        self.report_dir = self.project_root / 'Artifacts' / 'reports'
        self.report_data = {"before_transformation": {}, "after_transformation": {}, "summary": {}}
        self.schema_config = self._load_schema()
        self.tables_config = self.schema_config.get('raw', {})
        self.dataframes = {}

        
    def _load_schema(self):
        try:
            with open(self.schema_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.exception("Lỗi khi đọc file schema:")
            raise DataPipelineException(str(e), sys)

    # -------------------------------------------------------------------
    # MỚI BỔ SUNG: Hàm cập nhật thông tin schema staging vào file YAML
    # -------------------------------------------------------------------
    def _update_staging_schema_report(self, staging_metadata: dict):
        """
        Đọc file tables_created_report.yaml hiện tại, thêm/cập nhật 
        thông tin schema 'staging' và ghi đè lại file.
        """
        try:
            # Đọc lại config mới nhất từ file
            current_config = self._load_schema()
            
            # Khởi tạo key 'staging' nếu chưa có
            if 'staging' not in current_config:
                current_config['staging'] = {}
                
            # Cập nhật metadata của các bảng mới vào schema staging
            current_config['staging'].update(staging_metadata)
            
            # Ghi xuống YAML
            with open(self.schema_path, 'w', encoding='utf-8') as file:
                yaml.dump(current_config, file, allow_unicode=True, default_flow_style=False, sort_keys=False)
                
            logger.info(f"Đã cập nhật cấu trúc bảng staging vào: {self.schema_path}")
        except Exception as e:
            logger.exception("Lỗi khi ghi lịch sử cập nhật bảng vào file YAML:")
            raise DataPipelineException(str(e), sys)

    def _save_transformation_report(self):
        try:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.report_dir / f"caloriePrediction_transformation_report_{timestamp}.yaml"
            with open(report_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.report_data, f, allow_unicode=True, default_flow_style=False)
            logger.info("Đã xuất report transformation tại: %s", report_path)
        except Exception as e:
            logger.exception("Lỗi khi lưu report:")
            raise DataPipelineException(str(e), sys)

    def _get_date_column_name(self, table_name):
        config = self.tables_config.get(table_name, {})
        columns_info = config.get('columns', {})
        
        possible_date_keywords = ['activitydate', 'date', 'activityhour', 'activityminute', 'time', 'activity_date']
        for col_name in columns_info.keys():
            if col_name.lower() in possible_date_keywords:
                return col_name
                
        # Fallback check trực tiếp trên df
        if table_name in self.dataframes:
            for col in self.dataframes[table_name].columns:
                if col.lower() in possible_date_keywords:
                    return col
        return None

    def execute_pipeline(self):
        try:
            logger.info("Bắt đầu Data Transformation Pipeline...")
            
            # Phase 1: Tải dữ liệu và chuẩn hóa trục thời gian thành cột 'date'
            self.phase_1_ingestion_and_standardize()
            
            # Phase 2: Feature Engineering (Đồng bộ khung thời gian, Aggregate & Trám Weight)
            transformed_dfs = self.phase_2_feature_engineering_and_alignment()
            
            # Phase 3: Lưu các bảng trung gian vào schema staging
            self.phase_3_save_to_staging(transformed_dfs)
            
            # Phase 4: Master Join và Cleanup
            final_df = self.phase_4_integration_and_cleaning(transformed_dfs)

            # Metadata Report
            self.report_data["after_transformation"]["final_dataset"] = {
                "rows": int(final_df.shape[0]),
                "columns": int(final_df.shape[1]),
                "missing_values": int(final_df.isna().sum().sum()),
                "memory_usage_mb": float(final_df.memory_usage(deep=True).sum() / (1024 ** 2))
            }
            
            total_raw_rows = sum(t["rows"] for t in self.report_data["before_transformation"].values())
            self.report_data["summary"] = {
                "total_raw_rows": total_raw_rows,
                "final_rows": int(final_df.shape[0]),
                "retention_rate": f"{(final_df.shape[0] / total_raw_rows * 100):.2f}%" if total_raw_rows > 0 else "0%"
            }
            self._save_transformation_report()
            logger.info("Hoàn thành Data Transformation Pipeline!")
            return final_df
        except Exception as e:
            logger.exception("Pipeline thất bại:")
            raise DataPipelineException(str(e), sys)

    # ==========================================
    # PHASE 1: INGESTION & CHUẨN HÓA
    # ==========================================
    def phase_1_ingestion_and_standardize(self):
        logger.info("Phase 1: Ingestion & Standardize (Chuẩn hóa cột mốc thời gian thành 'date')...")
        with get_connection() as conn:
            for table_name in self.tables_config.keys():
                try:
                    date_col = self._get_date_column_name(table_name)
                    if date_col:
                        query = f"""
                            COPY (SELECT *, CAST("{date_col}" AS TIMESTAMP) AS date_parsed 
                            FROM raw."{table_name}") TO STDOUT WITH CSV HEADER
                        """
                    else:
                        query = f'COPY raw."{table_name}" TO STDOUT WITH CSV HEADER'
                        
                    buffer = io.BytesIO()
                    with conn.cursor() as cur:
                        with cur.copy(query) as copy:
                            for data in copy:
                                buffer.write(data)
                                
                    buffer.seek(0)
                    df = pd.read_csv(buffer, low_memory=False)

                    # Ghi nhận Report
                    self.report_data["before_transformation"][table_name] = {
                        "rows": int(df.shape[0]), "columns": int(df.shape[1]),
                        "missing_values": int(df.isna().sum().sum()),
                        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / (1024 ** 2))
                    }

                    if date_col and 'date_parsed' in df.columns:
                        if date_col in df.columns:
                            df = df.drop(columns=[date_col])
                        df.rename(columns={'date_parsed': 'date'}, inplace=True)
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
                            
                    self.dataframes[table_name] = df
                    logger.info("Đã load bảng: %s", table_name)
                except Exception as e:
                    logger.exception("Không thể đọc bảng %s từ PostgreSQL:", table_name)
                    raise DataTransformationError(str(e), sys)

    # ==========================================
    # PHASE 2: FEATURE ENGINEERING & ALIGNMENT
    # ==========================================
    def phase_2_feature_engineering_and_alignment(self):
        try:
            logger.info("Phase 2: Feature Engineering & Alignment...")
            transformed_dfs = {}
            n_workers = max(1, multiprocessing.cpu_count() - 1)
            
            # Lấy df dailyActivity làm bảng mốc chuẩn
            daily_activity_key = next((k for k in self.dataframes.keys() if 'dailyactivity_merged' in k.lower()), None)
            weight_key = next((k for k in self.dataframes.keys() if 'weightloginfo_merged' in k.lower()), None)
            
            if not daily_activity_key:
                raise DataTransformationError("Không tìm thấy bảng dailyActivity làm mốc chuẩn (spine).", sys)
            
            daily_df = self.dataframes[daily_activity_key]
            transformed_dfs[daily_activity_key] = daily_df # Giữ nguyên daily_df

            # Xương sống thời gian (Base grid) cho toàn bộ user
            base_grid = daily_df[['Id', 'date']].drop_duplicates().sort_values(by=['Id', 'date'])

            # 1. Xử lý các bảng có dữ liệu phân giải cao (hourly, intensities, heartrate...)
            for table_name, df in self.dataframes.items():
                if table_name == daily_activity_key or table_name == weight_key:
                    continue
                
                if 'date' in df.columns and 'Id' in df.columns:
                    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    num_cols = [col for col in num_cols if col != 'Id']
                    
                    if num_cols:
                        logger.info(f"Aggregating bảng chi tiết: {table_name}")
                        unique_ids = df['Id'].unique()
                        id_chunks = np.array_split(unique_ids, n_workers)
                        
                        chunks = []
                        for id_chunk in id_chunks:
                            df_chunk = df[df['Id'].isin(id_chunk)]
                            chunks.append((df_chunk, num_cols, table_name))
                            
                        with ProcessPoolExecutor(max_workers=n_workers) as executor:
                            results = list(executor.map(process_agg_chunk, chunks))
                            
                        grouped_df = pd.concat(results, ignore_index=True)
                        transformed_dfs[table_name] = grouped_df

            # 2. Xử lý bảng Weight Log (Dữ liệu thưa thớt)
            if weight_key:
                logger.info("Xử lý bảng Weight: Tạo base ngày và điền giá trị (ffill, bfill)...")
                weight_df = self.dataframes[weight_key]
                # Merge weight_df vào base_grid bằng Left Join để sinh ra các ngày bị thiếu
                merged_weight = pd.merge(base_grid, weight_df, on=['Id', 'date'], how='left')
                merged_weight = merged_weight.sort_values(by=['Id', 'date'])
                
                # Điền khuyết các cột theo Id: Lấy giá trị của ngày gần nhất trước đó (ffill) rồi lùi (bfill)
                merged_weight = merged_weight.groupby('Id', group_keys=False).apply(lambda x: x.ffill().bfill())
                transformed_dfs[weight_key] = merged_weight

            return transformed_dfs

        except Exception as e:
            logger.exception("Lỗi Phase 2:")
            raise DataTransformationError(str(e), sys)

    # ==========================================
    # PHASE 3: STAGING & SAVE POSTGRESQL (ĐÃ CẬP NHẬT GHI YAML)
    # ==========================================
    def phase_3_save_to_staging(self, transformed_dfs):
        logger.info("Phase 3: Lưu các bảng sau khi biến đổi vào schema 'staging'...")
        staging_metadata = {}
        try:
            # Tạo schema staging nếu chưa tồn tại
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE SCHEMA IF NOT EXISTS staging;")
                conn.commit()

            engine = get_engine()  # Lấy engine SQLAlchemy từ connection.py
            
            for table_name, df in transformed_dfs.items():
                target_table = f"{table_name}_transformed"
                logger.info(f"Đang ghi bảng {target_table} vào schema staging...")
                
                # 1. Lưu vào PostgreSQL
                df.to_sql(
                    name=target_table, 
                    con=engine, 
                    schema='staging', 
                    if_exists='replace', 
                    index=False,
                    chunksize=5000,
                    method='multi'
                )
                
                # 2. Thu thập metadata định dạng bảng cho file YAML
                # Convert object types thành strings để YAML dễ dàng dump
                columns_info = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
                
                staging_metadata[target_table] = {
                    "created_at": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "rows": int(df.shape[0]),
                    "columns_count": int(df.shape[1]),
                    "columns": columns_info
                }
                
            logger.info("Lưu các bảng staging vào Database thành công.")
            
            # 3. Ghi thông tin metadata vào docs/architecture/tables_created_report.yaml
            self._update_staging_schema_report(staging_metadata)
            
        except Exception as e:
            logger.exception("Lỗi khi lưu bảng vào staging PostgreSQL hoặc ghi schema report:")
            raise DataTransformationError(str(e), sys)

    # ==========================================
    # PHASE 4: INTEGRATION & LÀM SẠCH
    # ==========================================
    def phase_4_integration_and_cleaning(self, transformed_dfs):
        logger.info("Phase 4: Hợp nhất các bảng với dailyActivity (Master Join)...")
        try:
            daily_activity_key = next((k for k in transformed_dfs.keys() if 'dailyactivity_merged' in k.lower()), None)
            master_df = transformed_dfs[daily_activity_key]
            
            # Left join các bảng aggregated và bảng weight (đã nội suy) vào bảng master dựa trên Id và date
            for table_name, df in transformed_dfs.items():
                if table_name == daily_activity_key:
                    continue
                logger.info(f"Đang Left Join bảng {table_name} vào Master Dataframe...")
                master_df = pd.merge(master_df, df, on=['Id', 'date'], how='left')

            # Xử lý missing values sau khi join
            # Có thể có những Id tồn tại ở daily_df nhưng hoàn toàn không có trong bảng phụ
            # -> Các cột aggregation sẽ là NaN -> Fill 0 hoặc median tùy nghiệp vụ
            master_df = master_df.fillna(0) # Tạm thời fill 0, bạn có thể customize phần này
            
            # Bỏ các cột trùng lặp hoặc không cần thiết nếu có
            master_df = master_df.loc[:,~master_df.columns.duplicated()]
            
            logger.info(f"Master Dataset shape sau khi hợp nhất: {master_df.shape}")
            return master_df

        except Exception as e:
            logger.exception("Lỗi Phase 4:")
            raise DataTransformationError(str(e), sys)

    # ==========================================
    # PHASE 4: TỐI ƯU HÓA & ĐÓNG GÓI
    # ==========================================
    def phase_4_optimization_and_packaging(self, master_df):
        logger.info("Phase 4: Optimization & Packaging...")
        try:
            logger.info("Phase 4: Lọc đa cộng tuyến thông minh (So sánh với Target)...")
            target_col = 'Calories'
            
            numeric_df = master_df.select_dtypes(include=[np.number])
            
            # 1. Tính ma trận tương quan toàn cục
            corr_matrix = numeric_df.corr().abs()
            
            # 2. Lấy riêng mức độ tương quan của từng biến đối với cột Target ('Calories')
            target_corr = corr_matrix[target_col]
            
            # 3. Tạo ma trận tương quan chỉ giữa các features
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
            df_final = master_df.drop(columns=to_drop)
            
            logger.info("Đã so sánh với Target và xóa %s cột bị đa cộng tuyến (>0.85).", len(to_drop))
            
            # 5. Đóng gói ra thư mục Artifacts (DYNAMIC PATH)
            output_dir = self.project_root / "Data" / "processed"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Lưu Parquet (Khuyên dùng cho Model Training vì nhẹ và load nhanh)
            parquet_path = output_dir / "core_daily_calories_features.parquet"
            df_final.to_parquet(parquet_path, index=False)
            
            # Lưu CSV (Để dễ dàng xem bằng Excel/VSCode)
            csv_path = output_dir / "core_daily_calories_features.csv"
            df_final.to_csv(csv_path, index=False)

            # TỰ ĐỘNG SINH FEATURES.YAML
            features_list = df_final.columns.tolist()
            features_config = {
                "target": "Calories",
                "features": [col for col in features_list if col not in ['Id', 'Calories', 'date']],
                "metadata": {
                    "num_features": len(features_list) - 3,
                    "description": "Generated automatically after collinearity removal."
                }
            }

            # DYNAMIC PATH CHO CONFIG
            config_path = self.project_root / "config" / "pipelines" / "ml_calories" / "features.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(features_config, f, allow_unicode=True, default_flow_style=False)

            logger.info("Đã xuất cấu hình features phục vụ training & inference tại: %s", config_path)
            logger.info("Đã xuất file Final Dataset (Parquet & CSV) sẵn sàng cho Model tại: %s", output_dir)
            
            # --------------------------------------------------
            # KHUNG LỆNH ĐẨY LÊN DATABASE (POSTGRESQL CORE)
            # --------------------------------------------------
            # with get_connection() as conn:
            #     df_final.to_sql(name="daily_calories_features", con=conn, schema="core", if_exists="replace", index=False)
            # logger.info("# [DB] Đã push bảng core.daily_calories_features lên PostgreSQL")
            
            # TRẢ VỀ DF_FINAL, KHÔNG PHẢI MASTER_DF
            return df_final
            
        except Exception as e:
            logger.exception("Lỗi tối ưu hóa đa cộng tuyến và Đóng gói:")
            raise DataTransformationError(str(e), sys)

    def preprocess_core_daily_calories(self, df, exclude_cat_col='categorical_id', poly_features=['low_corr_1', 'low_corr_2'], missing_thresh=0.5, zero_thresh=0.5):
        """
        Tiền xử lý dữ liệu: Loại bỏ cột/hàng nhiễu (>50% missing/zero) và xử lý null.
        Tích hợp vào DataTransformationPipeline.
        """
        processed_df = df.copy()

        # ==========================================
        # 1. XÓA CỘT NHIỄU (Missing/Zero > 50%)
        # ==========================================
        missing_percentages = processed_df.isnull().mean()
        cols_to_drop_missing = missing_percentages[missing_percentages > missing_thresh].index.tolist()

        zero_percentages = (processed_df == 0).mean()
        cols_to_drop_zero = zero_percentages[zero_percentages > zero_thresh].index.tolist()

        cols_to_drop = set(cols_to_drop_missing + cols_to_drop_zero)

        # BẢO VỆ DỮ LIỆU CỐT LÕI: Giữ lại biến đa thức và mã định danh
        cols_to_drop = [col for col in cols_to_drop if col not in poly_features]
        if exclude_cat_col in cols_to_drop:
            cols_to_drop.remove(exclude_cat_col)

        processed_df.drop(columns=list(cols_to_drop), inplace=True, errors='ignore')

        # ==========================================
        # 2. XÓA HÀNG NHIỄU (Missing > 50%)
        # ==========================================
        # Tham số thresh trong dropna yêu cầu số lượng giá trị non-null tối thiểu.
        # Để xóa hàng có > 50% missing, ta cần giữ lại các hàng có >= 50% dữ liệu hợp lệ.
        min_valid_cols = len(processed_df.columns) * (1 - missing_thresh)
        processed_df.dropna(axis=0, thresh=min_valid_cols, inplace=True)

        # ==========================================
        # 3. ĐIỀN GIÁ TRỊ THIẾU CHO PHẦN CÒN LẠI
        # ==========================================
        numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
        
        # Điền null bằng median (bỏ qua cột định danh)
        for col in numeric_cols:
            if col != exclude_cat_col:
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                
        # Điền null bằng mode cho cột định danh
        if exclude_cat_col in processed_df.columns:
            processed_df[exclude_cat_col] = processed_df[exclude_cat_col].fillna(processed_df[exclude_cat_col].mode()[0])

        return processed_df


if __name__ == "__main__":
    pipeline = DataTransformationPipeline()
    final_dataset = pipeline.execute_pipeline()