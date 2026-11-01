import os
import pandas as pd
from src.nutrition_core.logging.logger import logger, monitor_performance
from src.nutrition_core.pandera.evidently_engine import EvidentlyDriftEngine
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker

class CaloriesDriftEvaluator:
    def __init__(self, config: dict, tracker: MLOpsTracker):
        """
        Khởi tạo Drift Evaluator.
        config (dict): Cấu hình tổng (hoặc cấu hình evidently_profile) được load từ Orchestrator.
        tracker (MLOpsTracker): Instance của tracker để duy trì cùng một run_id trên MLflow.
        """
        self.config = config
        self.tracker = tracker

    @monitor_performance()
    def evaluate_drift(self, current_df: pd.DataFrame) -> bool:
        """
        Thực thi đánh giá Data Drift. 
        current_df được truyền trực tiếp từ module Data Ingestion, không kết nối lại DB ở đây.
        """
        logger.info("Bắt đầu quy trình đánh giá Data Drift (Evidently AI)...")
        
        # 1. Nạp Reference Data từ cấu hình
        ref_path = self.config.get('reference_data_path', 'data/reference_data.csv')
        if not os.path.exists(ref_path):
            logger.error(f"Không tìm thấy file Reference Data tại: {ref_path}")
            raise FileNotFoundError(f"Missing reference data: {ref_path}")
            
        logger.info(f"Nạp Reference Data từ: {ref_path}")
        reference_df = pd.read_csv(ref_path)
        
        # 2. Khởi tạo Engine với các tham số từ cấu hình
        project_name = self.config.get('project_name', 'Calories_Drift_Project')
        artifact_dir = self.config.get('artifact_dir', 'Artifacts/reports/ml_calories/evidently')
        
        drift_engine = EvidentlyDriftEngine(
            project_name=project_name,
            artifact_dir=artifact_dir,
            tracker=self.tracker # Truyền tracker vào engine để đẩy metrics lên MLflow
        )
        
        # 3. Lấy tham số test và mapping cột
        options = self.config.get('options', {})
        column_mapping = self.config.get('column_mapping', {})
        
        # 4. Thực thi đánh giá
        logger.info("Tiến hành đối chiếu Current Data và Reference Data...")
        drift_detected = drift_engine.run_drift_detection(
            reference_df=reference_df,
            current_df=current_df,
            options=options,
            column_mapping=column_mapping
        )
        
        # 5. Đánh giá kết quả cuối cùng
        if drift_detected:
            logger.warning("⚠️ CẢNH BÁO: Phát hiện Data Drift! Phân phối dữ liệu đã thay đổi.")
        else:
            logger.info("✅ Dữ liệu ổn định, không có dấu hiệu Data Drift.")
            
        return drift_detected