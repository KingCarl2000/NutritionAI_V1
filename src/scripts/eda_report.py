import sys
import json
import numpy as np
from pathlib import Path
import yaml
import pandas as pd
from sqlalchemy import create_engine
from ydata_profiling import ProfileReport

# 1. Thiết lập đường dẫn Root để import các module trong src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Import config đã được thiết lập sẵn trong dự án
from src.postgres.core.db_config import config

# Cấu hình đường dẫn bằng Pathlib
SCHEMA_PATH = PROJECT_ROOT / "src" / "api" / "data_schema" / "schema.yaml"
ARTIFACTS_DIR = PROJECT_ROOT / "Artifacts"

# Đảm bảo thư mục Artifacts luôn tồn tại
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Khởi tạo kết nối cơ sở dữ liệu từ db_config
db_url = f"postgresql+psycopg2://{config.user}:{config.password}@{config.host}:{config.port}/{config.dbname}"
engine = create_engine(db_url)

def safe_float(val):
    """Hàm hỗ trợ chuyển đổi an toàn các giá trị numpy sang python float/null"""
    if pd.isna(val) or np.isinf(val):
        return None
    return float(val)

def safe_int(val):
    """Hàm hỗ trợ chuyển đổi an toàn sang python int"""
    if pd.isna(val) or np.isinf(val):
        return None
    return int(val)

def calculate_custom_metrics(df):
    """
    Hàm tính toán chính xác các chỉ số phân tích theo yêu cầu:
    1. Table-level, 2. Numerical, 3. Categorical/Datetime, 4. Relationships
    """
    metrics = {
        "1_table_level_metrics": {},
        "2_numerical_metrics": {},
        "3_categorical_and_datetime_metrics": {},
        "4_relationships": {}
    }
    
    total_rows = len(df)
    total_cols = len(df.columns)
    if total_rows == 0:
        return metrics

    # 1. Chỉ số tổng quan cấp độ bảng (Table-level Metrics)
    total_cells = total_rows * total_cols
    missing_cells = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    
    metrics["1_table_level_metrics"] = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "duplicate_rows_count": duplicate_rows,
        "duplicate_rows_ratio": safe_float(duplicate_rows / total_rows),
        "missing_cells_ratio": safe_float(missing_cells / total_cells) if total_cells > 0 else 0
    }

    # Phân loại cột
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number', 'datetime', 'datetimetz']).columns.tolist()

    # 2. Chỉ số thống kê cho cột dạng số (Numerical Metrics)
    for col in num_cols:
        col_data = df[col].dropna()
        metrics["2_numerical_metrics"][col] = {
            "missing_null_ratio": safe_float(df[col].isnull().mean()),
            "zero_ratio": safe_float((df[col] == 0).mean()),
            "min": safe_float(col_data.min()),
            "max": safe_float(col_data.max()),
            "mean": safe_float(col_data.mean()),
            "median_50": safe_float(col_data.median()),
            "std_dev": safe_float(col_data.std()),
            "iqr_25_75": safe_float(col_data.quantile(0.75) - col_data.quantile(0.25)) if not col_data.empty else None,
            "skewness": safe_float(col_data.skew()),
            "kurtosis": safe_float(col_data.kurt())
        }

    # 3. Chỉ số cho cột Phân loại (Categorical)
    for col in cat_cols:
        mode_val = df[col].mode()
        metrics["3_categorical_and_datetime_metrics"][col] = {
            "type": "categorical",
            "missing_null_ratio": safe_float(df[col].isnull().mean()),
            "distinct_unique_count": safe_int(df[col].nunique()),
            "mode": str(mode_val[0]) if not mode_val.empty else None,
            "frequency": safe_int((df[col] == mode_val[0]).sum()) if not mode_val.empty else None
        }

    # 3. Chỉ số cho cột Thời gian (Datetime)
    # Tự động cố gắng chuyển đổi cột object sang datetime nếu có thể (để bắt được các chuỗi thời gian)
    for col in cat_cols.copy():
        if df[col].dtype == 'object':
            try:
                # Kiểm tra nhẹ xem có giống định dạng ngày tháng không
                if not df[col].dropna().empty and isinstance(df[col].dropna().iloc[0], str) and ('-' in df[col].dropna().iloc[0] or '/' in df[col].dropna().iloc[0]):
                    parsed_date = pd.to_datetime(df[col], errors='coerce')
                    if parsed_date.notnull().sum() > (len(df) * 0.5): # Nếu hơn 50% là ngày tháng hợp lệ
                        df[col] = parsed_date
                        datetime_cols.append(col)
                        cat_cols.remove(col)
                        # Cập nhật lại loại trong dictionary
                        metrics["3_categorical_and_datetime_metrics"].pop(col, None)
            except:
                pass

    for col in datetime_cols:
        mode_val = df[col].mode()
        metrics["3_categorical_and_datetime_metrics"][col] = {
            "type": "datetime",
            "missing_null_ratio": safe_float(df[col].isnull().mean()),
            "distinct_unique_count": safe_int(df[col].nunique()),
            "mode": str(mode_val[0]) if not mode_val.empty else None,
            "min_date": str(df[col].min()) if not pd.isna(df[col].min()) else None,
            "max_date": str(df[col].max()) if not pd.isna(df[col].max()) else None
        }

    # 4. Ma trận tương quan (Relationships)
    if len(num_cols) > 1:
        corr_data = df[num_cols].dropna(how='all')
        
        pearson_matrix = corr_data.corr(method='pearson').round(4)
        spearman_matrix = corr_data.corr(method='spearman').round(4)
        
        # Chuyển đổi DataFrame thành dictionary lồng nhau, thay thế NaN bằng None
        metrics["4_relationships"] = {
            "pearson_correlation": pearson_matrix.where(pd.notnull(pearson_matrix), None).to_dict(),
            "spearman_correlation": spearman_matrix.where(pd.notnull(spearman_matrix), None).to_dict()
        }
    else:
        metrics["4_relationships"] = {
            "message": "Không đủ số lượng cột dạng số (tối thiểu 2) để tính toán ma trận tương quan."
        }

    return metrics

def generate_reports():
    if not SCHEMA_PATH.exists():
        print(f"❌ Không tìm thấy file schema tại: {SCHEMA_PATH}")
        return

    # Đọc file định nghĩa schema
    with open(SCHEMA_PATH, 'r', encoding='utf-8') as file:
        schema_data = yaml.safe_load(file)

    if not schema_data:
        print("Cảnh báo: File schema trống.")
        return

    # Quét từng bảng và tạo báo cáo
    for table_name, table_info in schema_data.items():
        db_schema = table_info.get('db_schema', 'raw')
        print(f"\n🔄 Đang tiến hành lấy mẫu và tạo báo cáo cho: {db_schema}.{table_name}...")
        
        # Lấy mẫu 10,000 dòng để tránh lỗi tràn bộ nhớ (OOM)
        query = f'SELECT * FROM {db_schema}."{table_name}" LIMIT 10000'
        
        try:
            # Đọc dữ liệu vào DataFrame
            df = pd.read_sql(query, engine)
            
            # Bỏ qua nếu bảng trống
            if df.empty:
                print(f"⚠️ Bỏ qua {table_name}: Bảng không có dữ liệu.")
                continue

            # --- 1. XUẤT BÁO CÁO CẤU TRÚC JSON (MỚI) ---
            print("  -> Đang tính toán các chỉ số pipeline tùy chỉnh (JSON)...")
            custom_metrics = calculate_custom_metrics(df.copy())
            json_output_file = ARTIFACTS_DIR / f"{table_name}_metrics.json"
            
            with open(json_output_file, 'w', encoding='utf-8') as f:
                json.dump(custom_metrics, f, ensure_ascii=False, indent=4)
            print(f"  ✅ Đã lưu file cấu trúc: {json_output_file.name}")

            # --- 2. XUẤT BÁO CÁO TRỰC QUAN HTML (GIỮ NGUYÊN) ---
            print("  -> Đang khởi tạo báo cáo trực quan ydata-profiling (HTML)...")
            profile = ProfileReport(
                df, 
                title=f"EDA Dataset Schema - {table_name}",
                explorative=True 
            )
            html_output_file = ARTIFACTS_DIR / f"{table_name}_dataset_schema.html"
            profile.to_file(html_output_file)
            print(f"  ✅ Đã lưu file trực quan: {html_output_file.name}")
            
        except Exception as e:
            print(f"❌ LỖI: Không thể xử lý bảng {table_name}. Lỗi: {e}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print(" BẮT ĐẦU TIẾN TRÌNH TẠO BÁO CÁO EDA TỰ ĐỘNG ")
    print("="*50)
    generate_reports()
    print("\n" + "="*50)
    print(" HOÀN TẤT TIẾN TRÌNH ")
    print("="*50)