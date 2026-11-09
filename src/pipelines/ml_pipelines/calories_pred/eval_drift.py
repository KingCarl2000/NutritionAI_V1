import os
import yaml
import pandas as pd
from typing import Tuple, Dict, Optional

from src.nutrition_core.logging.logger import logger, monitor_performance
from src.nutrition_core.pandera.evidently_engine import EvidentlyDriftEngine

# =============================================================================
# HELPER: LOAD YAML CONFIG
# =============================================================================
def load_yaml_config(path: str) -> dict:
    """Đọc file cấu hình YAML."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Không tìm thấy file cấu hình tại {path}")
        raise

# =============================================================================
# DRIFT EVALUATOR CLASS
# =============================================================================
class CaloriesDriftEvaluator:
    """
    Class đánh giá Data Drift tuân thủ nguyên tắc đọc tham số hoàn toàn từ file YAML cấu hình,
    không gán giá trị mặc định (hardcode) trong code.
    """
    def __init__(self, evidently_config: dict, global_config: dict):
        logger.info("Khởi tạo cấu hình cho Drift Evaluator...")
        self.evidently_config = evidently_config
        self.global_config = global_config
        
        # 1. Trích xuất Project Info
        self.project_name = self.evidently_config["engine_config"]["project_name"]
        
        # 2. Trích xuất Thresholds từ global.yaml
        drift_thresholds = self.global_config["mlops_thresholds"]["data_drift"]
        self.psi_warning = drift_thresholds["psi_warning"]
        self.psi_critical = drift_thresholds["psi_critical"]
        self.p_value_alpha = drift_thresholds["p_value_alpha"]
        
        # 3. Trích xuất Column Mapping
        self.column_mapping = self.evidently_config["column_mapping"]

    @monitor_performance(time_limit=10.0)
    def evaluate_drift(self, current_df: pd.DataFrame, reference_df: pd.DataFrame) -> Tuple[Dict, str]:
        """
        Đo lường độ trôi lệch dữ liệu và kiểm tra ngưỡng cảnh báo.
        """
        logger.info("Bắt đầu đánh giá Data Drift...")
        
        report_path = os.path.join(self.artifact_dir, "drift_report.html")
        metrics_path = os.path.join(self.artifact_dir, "drift_metrics.json")
        
        # Khởi tạo core engine của Evidently (tuân thủ API Registry)
        engine = EvidentlyDriftEngine(
            artifact_dir=self.artifact_dir,
            project_name=self.project_name
        )

        run_drift_report = getattr(engine, "run_drift_report")
        
        # Chạy report kết hợp truyền tham số p_value từ global.yaml
        drift_metrics = run_drift_report(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=self.column_mapping,
            report_path=report_path,
            metrics_path=metrics_path,
            p_value=self.p_value_alpha
        )
        
        # Lấy kết quả tỷ lệ Drift tổng quan trực tiếp từ dictionary
        drift_share = drift_metrics["dataset_drift_share"]
        
        # Kiểm định với Thresholds
        if drift_share >= self.psi_critical:
            logger.error(f"DATA DRIFT CRITICAL ALERT: Tỷ lệ drift ({drift_share:.2%}) "
                         f"vượt ngưỡng nghiêm trọng ({self.psi_critical:.2%}). Yêu cầu Retrain Model!")
        elif drift_share >= self.psi_warning:
            logger.warning(f"DATA DRIFT WARNING: Tỷ lệ drift ({drift_share:.2%}) "
                           f"vượt ngưỡng cảnh báo ({self.psi_warning:.2%}). Cần theo dõi!")
        else:
            logger.info(f"Dữ liệu ổn định (Drift = {drift_share:.2%}). Không vi phạm ngưỡng cảnh báo.")
            
        return drift_metrics, report_path

# =============================================================================
# WRAPPER FUNCTION
# =============================================================================
def run_drift_detection(reference_data: pd.DataFrame, current_data: pd.DataFrame, output_dir: Optional[str] = None) -> Tuple[Dict, str]:
    """
    Hàm Wrapper dùng để Data Validation Pipeline gọi thẳng vào mà không cần quan tâm chi tiết.
    """
    evaluator = CaloriesDriftEvaluator()
    return evaluator.evaluate_drift(current_df=current_data, reference_df=reference_data)

# =============================================================================
# MAIN EXECUTION (TESTING / LOCAL RUN)
# =============================================================================
if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. ĐỊNH NGHĨA TOÀN BỘ ĐƯỜNG DẪN TẠI ĐÂY (CONTROL PANEL)
    # ---------------------------------------------------------
    EVIDENTLY_YAML_PATH = "config/pipelines/ml_calories/evidently_profile.yaml"
    GLOBAL_YAML_PATH = "config/pipelines/global.yaml"
    
    # Input data paths (Thay đổi tuỳ ý cho các thí nghiệm khác nhau)
    REFERENCE_DATA_PATH = "Artifacts/data/ml_calories/processed/train.csv"
    CURRENT_DATA_PATH = "Artifacts/data/ml_calories/processed/current.csv"
    
    # Output report path
    REPORT_DIR = "Artifacts/reports/ml_calories/evidently"

    # ---------------------------------------------------------
    # 2. ĐỌC CONFIGS & DATA
    # ---------------------------------------------------------
    print("⏳ Đang tải file cấu hình...")
    with open(EVIDENTLY_YAML_PATH, "r", encoding="utf-8") as f:
        evidently_cfg = yaml.safe_load(f)
        
    with open(GLOBAL_YAML_PATH, "r", encoding="utf-8") as f:
        global_cfg = yaml.safe_load(f)

    print("⏳ Đang tải dữ liệu...")
    # NOTE: Bạn phải đảm bảo 2 file CSV này tồn tại, nếu không có thể thay bằng pd.DataFrame mock như trước
    try:
        ref_df = pd.read_csv(REFERENCE_DATA_PATH)
        curr_df = pd.read_csv(CURRENT_DATA_PATH)
    except FileNotFoundError as e:
        print(f"❌ Lỗi: {e}. Vui lòng kiểm tra lại đường dẫn file input.")
        exit(1)

    # ---------------------------------------------------------
    # 3. CHẠY PIPELINE
    # ---------------------------------------------------------
    print("🚀 Khởi chạy Pipeline Drift Evaluator...")
    pipeline = CaloriesDriftEvaluator(
        evidently_config=evidently_cfg, 
        global_config=global_cfg
    )
    
    metrics, report_file = pipeline.run(
        current_df=curr_df, 
        reference_df=ref_df, 
        report_output_dir=REPORT_DIR
    )
    
    print("\n✅ [HOÀN THÀNH]")
    print(f"📊 Dataset Drift Share: {metrics['dataset_drift_share']:.2%}")
    print(f"📁 Báo cáo chi tiết lưu tại: {report_file}")