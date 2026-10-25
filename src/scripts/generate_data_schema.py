import json
from pathlib import Path
from src.nutrition_core.logging.logger import logger

# Đường dẫn gốc của dự án (NutritionAI_V1)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =============================================================================
# HÀM XỬ LÝ LÕI (CORE LOGIC)
# =============================================================================

def generate_schema_skeleton(config: dict):
    """
    Sinh ra bộ khung (skeleton) Pandera schema dựa trên file JSON Metrics từ EDA.
    Để trống các ngưỡng sinh học (TODO) cho con người tự định nghĩa.
    """
    pipeline_name = config.get("pipeline_name")
    target_col = config.get("target_col")
    metrics_json_path = Path(config.get("metrics_json_path"))
    output_schema_path = Path(config.get("output_schema_path"))
    
    logger.info(f"=== Bắt đầu tạo khung Pandera Schema cho: '{pipeline_name}' ===")

    if not metrics_json_path.exists():
        logger.error(f"Không tìm thấy file EDA metrics tại {metrics_json_path}. Bỏ qua.")
        return

    try:
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    except Exception as e:
        logger.error(f"Lỗi khi đọc file JSON {metrics_json_path}: {e}")
        return
    
    # Trích xuất danh sách cột từ cấu trúc JSON của eda_report.py
    num_cols = metrics_data.get("2_numerical_metrics", {})
    cat_cols = metrics_data.get("3_categorical_and_datetime_metrics", {})
    
    # Chuẩn hóa tên class (vd: ml_calories -> MlCalories)
    model_name = "".join([word.capitalize() for word in pipeline_name.split("_")])
    input_class = f"{model_name}InputSchema"
    train_class = f"{model_name}TrainingSchema"

    # Tạo Header và Config
    lines = [
        "import pandera.pandas as pa",
        "from pandera.typing import Series",
        "",
        f"class {input_class}(pa.DataFrameModel):",
        "    class Config:",
        '        """Khung Schema sinh tự động từ EDA. CẦN CẬP NHẬT NGƯỠNG SINH HỌC."""',
        "        coerce = True",
        "        strict = True",
        ""
    ]

    # Khởi tạo các fields (Numerical mặc định là float, Categorical mặc định là str)
    # Kỹ sư dữ liệu sẽ tự điều chỉnh lại thành int nếu cần thông qua block TODO.
    for col_name in num_cols.keys():
        if col_name != target_col:
            lines.append(f"    {col_name}: Series[float] = pa.Field(nullable=False)  # TODO: Xác nhận lại kiểu (int/float) và cập nhật ngưỡng")

    for col_name in cat_cols.keys():
        if col_name != target_col:
            lines.append(f"    {col_name}: Series[str] = pa.Field(nullable=False)  # TODO: Cập nhật danh sách isin=[...]")

    # Kế thừa cho Training Schema (Nếu có biến mục tiêu)
    target_in_num = target_col in num_cols
    target_in_cat = target_col in cat_cols
    
    if target_in_num or target_in_cat:
        target_py_type = "float" if target_in_num else "str"
        lines.extend([
            "",
            f"class {train_class}({input_class}):",
            f"    {target_col}: Series[{target_py_type}] = pa.Field(nullable=False)  # TODO: Cập nhật ngưỡng cho Target"
        ])

    # Lưu ra file
    output_schema_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_schema_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    logger.info(f"Đã tạo thành công bộ khung schema tại: {output_schema_path}\n")

def run_all_schema_generations(pipelines_config: list):
    """Lặp qua danh sách cấu hình và thực thi tạo schema cho tất cả pipelines."""
    logger.info("Bắt đầu tiến trình tự động tạo Schema Skeleton từ EDA Metrics...")
    for config in pipelines_config:
        generate_schema_skeleton(config)
    logger.info("Hoàn tất tiến trình!")

# =============================================================================
# ĐIỀU PHỐI THỰC THI & CẤU HÌNH INPUT/OUTPUT
# =============================================================================
if __name__ == "__main__":
    # Cấu hình mapping 1-1 giữa file metrics JSON và file schema Python đầu ra
    ML_PIPELINES_CONFIG = [
        {
            "pipeline_name": "ml_calories",
            "target_col": "Calories",
            "metrics_json_path": PROJECT_ROOT / "Artifacts" / "EDA_reports" / "calories_metrics.json",
            "output_schema_path": PROJECT_ROOT / "config" / "pipelines" / "ml_calories" / "calories_schema.py",
        }
    ]
    
    run_all_schema_generations(ML_PIPELINES_CONFIG)