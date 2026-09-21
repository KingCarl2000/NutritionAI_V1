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

# Import connections and core
from src.postgres.core.connection import get_connection
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
    # Lưu ý: Hàm mode khá tốn tài nguyên so với các hàm vector hóa
    m = x.mode()
    return m.iloc[0] if not m.empty else np.nan

def get_skew(x):
    return x.skew()

def get_kurt(x):
    return x.kurt()

def process_agg_chunk(args):
    """
    Hàm Worker: Thực thi tính toán thống kê trên từng mảnh (chunk) dữ liệu.
    """
    df_chunk, num_cols, table_name = args
    
    agg_dict = {col: ['mean', 'max', 'min', 'std', 'median', 'var', get_mode, q1, q3, get_skew, get_kurt]
                for col in num_cols}
    
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
        # 1. Khởi tạo đường dẫn động (Dynamic Paths)
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        
        # Trỏ mặc định tới tables_created_report.yaml thay vì schema.yaml cũ
        if schema_path is None:
            self.schema_path = str(self.project_root / 'docs' / 'architecture' / 'tables_created_report.yaml')
        else:
            self.schema_path = schema_path
            
        self.report_dir = self.project_root / 'Artifacts' / 'reports'
        
        # 2. Khởi tạo cấu trúc lưu report
        self.report_data = {
            "before_transformation": {},
            "after_transformation": {},
            "summary": {}
        }
        
        # 3. Load config
        self.schema_config = self._load_schema()
        # Lấy thông tin các bảng từ dict 'raw' theo cấu trúc YAML mới
        self.tables_config = self.schema_config.get('raw', {})
        self.dataframes = {}

    def _load_schema(self):
        try:
            with open(self.schema_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.exception("Lỗi khi đọc file schema:")
            raise DataPipelineException(str(e), sys)
            
    def _save_transformation_report(self):
        """Hàm ghi file report ra Artifacts/reports"""
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
        
        cat_cols = []
        # Lấy các cột dựa trên yaml_type từ tables_created_report.yaml
        for col_name, col_attrs in columns_info.items():
            if col_attrs.get('yaml_type') in ['categorical', 'datetime', 'date']:
                cat_cols.append(col_name)

        possible_date_keywords = ['activitydate', 'date', 'activityhour', 'activityminute', 'time', 'activity_date']
        
        for col in cat_cols:
            if col.lower() in possible_date_keywords:
                return col

        if table_name in self.dataframes:
            for col in self.dataframes[table_name].columns:
                if col.lower() in possible_date_keywords:
                    return col
                    
        return None

    def execute_pipeline(self):
        try:
            logger.info("Bắt đầu Data Transformation Pipeline...")

            self.phase_1_ingestion_and_standardize()
            aggregated_dfs, weight_df = self.phase_2_feature_engineering()
            master_df = self.phase_3_integration_and_cleaning(aggregated_dfs, weight_df)
            final_df = self.phase_4_optimization_and_packaging(master_df)

            # Ghi nhận metadata sau khi biến đổi
            self.report_data["after_transformation"]["final_dataset"] = {
                "rows": int(final_df.shape[0]),
                "columns": int(final_df.shape[1]),
                "missing_values": int(final_df.isna().sum().sum()),
                "memory_usage_mb": float(final_df.memory_usage(deep=True).sum() / (1024 ** 2))
            }
            
            # Tổng kết report
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
    # PHASE 1: INGESTION & CHUẨN HÓA (CÓ REPORT)
    # ==========================================
    def phase_1_ingestion_and_standardize(self):
        logger.info("Phase 1: Chuẩn hóa trục thời gian và Load Data (Optimized via SQL Timestamp)...")
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

                    # Ghi nhận metadata vào Report (Trước biến đổi)
                    self.report_data["before_transformation"][table_name] = {
                        "rows": int(df.shape[0]),
                        "columns": int(df.shape[1]),
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
                    logger.info("Đã chuẩn hóa bảng: %s (Shape: %s)", table_name, df.shape)

                except Exception as e:
                    logger.exception("Không thể đọc bảng %s từ PostgreSQL:", table_name)
                    raise DataTransformationError(str(e), sys)

    # ==========================================
    # PHASE 2: FEATURE ENGINEERING (PARALLEL & DYNAMIC PATH)
    # ==========================================
    def phase_2_feature_engineering(self):
        try:
            logger.info("Phase 2: Bắt đầu Aggregation với xử lý song song (Chunking)...")
            aggregated_dfs = {}
            weight_df = None
            
            n_workers = max(1, multiprocessing.cpu_count() - 1)
            logger.info("Sử dụng %s CPU cores để tính toán song song.", n_workers)
            
            # Cập nhật đường dẫn lưu intermediate động
            inter_dir = self.project_root / 'Data' / 'intermediate'
            inter_dir.mkdir(parents=True, exist_ok=True)

            # Tìm key động, kiểm tra xem tên bảng CÓ CHỨA từ khóa hay không
            daily_activity_key = next((k for k in self.dataframes.keys() if 'dailyactivity_merged' in k.lower()), None)
            weight_key = next((k for k in self.dataframes.keys() if 'weightloginfo_merged' in k.lower()), None)

            for table_name, df in self.dataframes.items():
                if table_name == daily_activity_key:
                    continue
                elif table_name == weight_key:
                    weight_df = df
                else:
                    if 'date' in df.columns and 'Id' in df.columns:
                        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        num_cols = [col for col in num_cols if col != 'Id']
                        
                        if num_cols:
                            unique_ids = df['Id'].unique()
                            id_chunks = np.array_split(unique_ids, n_workers)
                            chunks = []
                            for id_chunk in id_chunks:
                                df_chunk = df[df['Id'].isin(id_chunk)]
                                chunks.append((df_chunk, num_cols, table_name))

                            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                                results = list(executor.map(process_agg_chunk, chunks))
                                logger.debug("Đang xử lý chunk cho %s user IDs", len(id_chunk))

                            grouped = pd.concat(results, ignore_index=True)
                            aggregated_dfs[table_name] = grouped
                            
                            # Lưu parquet trung gian bằng Path
                            parquet_path = inter_dir / f"{table_name}_agg.parquet"
                            grouped.to_parquet(parquet_path, index=False)
                        
                        logger.info("Đã Aggregate xong bảng: %s (Shape: %s)", table_name, grouped.shape)
                        
            return aggregated_dfs, weight_df
        except Exception as e:
            logger.exception("Lỗi Phase 2:")
            raise DataTransformationError(str(e), sys)

    # ==========================================
    # PHASE 3: INTEGRATION & LÀM SẠCH 
    # ==========================================
    def phase_3_integration_and_cleaning(self, aggregated_dfs, weight_df):
        logger.info("Phase 3: Integration & Cleaning...")
        try:
            logger.info("Phase 3: Master Join và Khảo sát dữ liệu...")
            
            # Tìm key động, kiểm tra xem tên bảng CÓ CHỨA từ khóa hay không
            daily_activity_key = next((k for k in self.dataframes.keys() if 'dailyactivity_merged' in k.lower()), None)
            if daily_activity_key is None:
                raise KeyError(f"Không tìm thấy bảng chứa 'dailyActivity_merged' trong dữ liệu. Các bảng hiện có: {list(self.dataframes.keys())}")
                
            master_df = self.dataframes[daily_activity_key].copy()
            
            # Merge các bảng Aggregate
            for table_name, agg_df in aggregated_dfs.items():
                master_df = pd.merge(master_df, agg_df, on=['Id', 'date'], how='left', suffixes=('', f'_{table_name}'))
                
            # Merge bảng Cân nặng
            if weight_df is not None:
                master_df = pd.merge(master_df, weight_df, on=['Id', 'date'], how='left')
                
            # ĐÁNH GIÁ DATAFRAME MỚI
            logger.info("---------- BÁO CÁO ĐÁNH GIÁ MASTER DATAFRAME ----------")
            
            # 1. Kích thước tập dữ liệu
            logger.info("[1] Kích thước (Shape): %s dòng, %s cột", master_df.shape[0], master_df.shape[1])
            
            # 2. Tính tỉ lệ phần trăm Missing Values
            missing_pct = (master_df.isnull().sum() / len(master_df)) * 100
            missing_cols = missing_pct[missing_pct > 0].sort_values(ascending=False)
            if not missing_cols.empty:
                logger.info("[2] Tỉ lệ Dữ liệu khuyết (Missing %%):\n%s", missing_cols.round(2).to_string())
            else:
                logger.info("[2] Tỉ lệ Dữ liệu khuyết: Không có.")
                
            # 3. Tính tỉ lệ phần trăm Giá trị 0
            zero_pct = (master_df == 0).sum() / len(master_df) * 100
            zero_cols = zero_pct[zero_pct > 0].sort_values(ascending=False)
            if not zero_cols.empty:
                logger.info("[3] Tỉ lệ Giá trị 0 (%%):\n%s", zero_cols.round(2).to_string())
            else:
                logger.info("[3] Tỉ lệ Giá trị 0: Không có.")
                
            logger.info("-------------------------------------------------------")

            # Tạo thư mục chứa báo cáo đánh giá (DYNAMIC PATH)
            report_dir = self.project_root / "Data" / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_file = report_dir / "caloriePrediction_data_assessment_report.txt"
            
            with open(report_file, "w", encoding="utf-8") as f:
                f.write("========== BÁO CÁO ĐÁNH GIÁ MASTER DATAFRAME ==========\n\n")
                f.write(f"1. Kích thước (Shape): {master_df.shape[0]} dòng, {master_df.shape[1]} cột\n\n")
                
                f.write("2. Tỉ lệ Dữ liệu khuyết (Missing %):\n")
                f.write(missing_cols.round(2).to_string() if not missing_cols.empty else "Không có missing value.")
                f.write("\n\n")
                
                f.write("3. Tỉ lệ Giá trị 0 (%):\n")
                f.write(zero_cols.round(2).to_string() if not zero_cols.empty else "Không có giá trị 0.")
                f.write("\n\n=======================================================")
            
            logger.info("Đã xuất báo cáo chi tiết ra file: %s", report_file)

            # Thực thi việc làm sạch dữ liệu
            logger.info("Thực thi làm sạch dữ liệu (Missing/Zero)...")
            master_df = self.preprocess_core_daily_calories(master_df)

            return master_df

        except Exception as e:
            logger.exception("Lỗi Integration và Khảo sát:")
            raise DataCleansingError(str(e), sys)

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