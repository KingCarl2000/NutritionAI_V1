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
    Class đánh giá Data Drift tích hợp đọc tham số động từ evidently_profile.yaml và global.yaml
    """
    def __init__(
        self, 
        evidently_config_path: str = "config/pipelines/ml_calories/evidently_profile.yaml",
        global_config_path: str = "config/pipelines/global.yaml"
    ):
        logger.info("Khởi tạo cấu hình cho Drift Evaluator...")
        self.evidently_config = load_yaml_config(evidently_config_path)
        self.global_config = load_yaml_config(global_config_path)
        
        # 1. Trích xuất I/O Paths từ evidently_profile.yaml
        paths = self.evidently_config.get("paths", {})
        self.ref_data_path = paths.get("reference_data", "Artifacts/data/ml_calories/processed/train.csv")
        self.artifact_dir = paths.get("artifact_dir", "Artifacts/reports/ml_calories/evidently")
        
        # 2. Trích xuất Thresholds từ global.yaml
        drift_thresholds = self.global_config.get("mlops_thresholds", {}).get("data_drift", {})
        self.psi_warning = drift_thresholds.get("psi_warning", 0.10)
        self.psi_critical = drift_thresholds.get("psi_critical", 0.25)
        self.p_value_alpha = drift_thresholds.get("p_value_alpha", 0.05)
        
        # 3. Trích xuất Column Mapping cho Evidently
        self.column_mapping = self.evidently_config.get("column_mapping", {})
        
        os.makedirs(self.artifact_dir, exist_ok=True)

    @monitor_performance(time_limit=10.0)
    def evaluate_drift(self, current_df: pd.DataFrame, reference_df: Optional[pd.DataFrame] = None) -> Tuple[Dict, str]:
        """
        Đo lường độ trôi lệch dữ liệu và kiểm tra ngưỡng cảnh báo.
        """
        logger.info("Bắt đầu đánh giá Data Drift...")
        
        # Load Reference Data nếu chưa được truyền vào
        if reference_df is None:
            if not os.path.exists(self.ref_data_path):
                raise FileNotFoundError(f"Missing reference data tại: {self.ref_data_path}")
            reference_df = pd.read_csv(self.ref_data_path)
        
        report_path = os.path.join(self.artifact_dir, "drift_report.html")
        metrics_path = os.path.join(self.artifact_dir, "drift_metrics.json")

        
        # Khởi tạo core engine của Evidently (tuân thủ API Registry)
        engine = EvidentlyDriftEngine(
            artifact_dir=self.artifact_dir,
            project_name=self.evidently_config.get("project_name", "Calories Drift Evaluation")
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
        
        # Lấy kết quả tỷ lệ Drift tổng quan (Tùy thuộc vào output của EvidentlyDriftEngine của bạn)
        # Ví dụ: Sử dụng share of drifted columns
        drift_share = drift_metrics.get("dataset_drift_share", 0.0)
        
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