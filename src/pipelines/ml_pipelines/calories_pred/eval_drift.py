import os
import datetime
import pandas as pd
from typing import Dict, Any

from src.nutrition_core.pandera.evidently_engine import EvidentlyDriftEngine
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker
from evidently.sdk.panels import PanelMetric, counter_panel, line_plot_panel # Cho Dashboard UI

class CaloriesDriftEvaluator:
    """
    Module Pipeline đánh giá Data Drift cho bài toán Calories Prediction.
    Sử dụng EvidentlyDriftEngine làm lõi xử lý trung tâm.
    """
    
    def __init__(self, config: Dict[str, Any], mlops_tracker: MLOpsTracker = None):
        self.config = config
        self.tracker = mlops_tracker
        self.logger = logger
        
        # [Nhiệm vụ 4] Khởi tạo Engine (Đã bao gồm kết nối Local/Cloud Workspace)
        self.engine = EvidentlyDriftEngine(
            artifact_dir=self.config.get("artifact_dir", "Artifacts/reports/ml_calories/data_drift"),
            project_name=self.config.get("project_name", "Calories_Prediction_Drift"),
            evidently_workspace_url=self.config.get("workspace_url", "Artifacts/evidently_workspace"),
            mlops_tracker=self.tracker
        )

    def _prepare_drift_options(self) -> Dict[str, Any]:
        """
        [Nhiệm vụ 2]: Cấu hình DataDriftPreset từ YAML.
        Ánh xạ các thông số drift_share, stattest, num_stattest, cat_stattest, threshold...
        """
        drift_cfg = self.config.get("drift_options", {})
        
        return {
            "dataset_drift_share": drift_cfg.get("drift_share", 0.5),
            "default_method": drift_cfg.get("stattest", None),
            "p_value_alpha": drift_cfg.get("threshold", None),
            "numerical": {
                "method": drift_cfg.get("num_stattest", None),
                "threshold": drift_cfg.get("num_threshold", None)
            },
            "categorical": {
                "method": drift_cfg.get("cat_stattest", None),
                "threshold": drift_cfg.get("cat_threshold", None)
            },
            "text": {
                "method": drift_cfg.get("text_stattest", None),
                "threshold": drift_cfg.get("text_threshold", None)
            },
            "per_feature_stattest": drift_cfg.get("per_column_stattest", None),
            "columns_to_check_drift": drift_cfg.get("columns", [])
        }

    def execute_evaluation(self, current_df: pd.DataFrame, reference_df: pd.DataFrame):
        """
        Thực thi toàn bộ luồng Drift Detection, Export và Monitoring.
        """
        self.logger.info("Bắt đầu quy trình đánh giá Data Drift...")
        
        # [Nhiệm vụ 1]: Chuẩn hóa & Khai báo dữ liệu (DataDefinition)
        # engine sẽ tự nạp numerical_features, categorical_features, target, task...
        column_mapping_cfg = self.config.get("column_mapping", {})
        
        # Lấy cấu hình Drift (Nhiệm vụ 2) và các cấu hình khác
        drift_options_cfg = self._prepare_drift_options()
        data_quality_options = self.config.get("data_quality_options", {})
        regression_options = self.config.get("regression_options", {})
        test_suite_options = self.config.get("test_suite_options", {})

        # Gọi Engine để tạo Report, chạy TestSuite và đẩy (add_report) lên Workspace
        engine_result = self.engine.run_drift_detection(
            current_df=current_df,
            reference_df=reference_df,
            column_mapping_cfg=column_mapping_cfg,
            drift_options_cfg=drift_options_cfg,
            data_quality_options=data_quality_options,
            regression_options=regression_options,
            test_suite_options=test_suite_options
        )
        
        report = engine_result.get("report")
        test_suite = engine_result.get("test_suite")

        # [Nhiệm vụ 3]: Xuất báo cáo Đa định dạng (Multiformat Export)
        artifact_dir = self.engine.artifact_dir
        if report:
            # 1. HTML Export
            report.save_html(os.path.join(artifact_dir, "drift_report.html"))
            # 2. JSON Export
            report.save_json(os.path.join(artifact_dir, "drift_report.json"))
            # 5. Full Snapshot Serialization (.json thay vì get_json)
            report.save(os.path.join(artifact_dir, "drift_snapshot.json"))
            
            # 3. Python Dict Export (Chuyển đổi trên RAM)
            report_dict = report.as_dict()
            
            # 4. Pandas DataFrame Export (Chuyển list metrics thành CSV/Dataframe)
            try:
                metrics_data = report_dict.get("metrics", [])
                if metrics_data:
                    pd.DataFrame(metrics_data).to_csv(os.path.join(artifact_dir, "drift_metrics.csv"), index=False)
            except Exception as e:
                self.logger.warning(f"⚠️ Lỗi khi xuất Pandas DataFrame: {e}")
                
            self.logger.info(f"✓ Đã xuất 5 định dạng Report (HTML, JSON, Dict, DataFrame, Snapshot) tại {artifact_dir}")
        else:
            report_dict = {}

        if test_suite:
            test_suite.save_html(os.path.join(artifact_dir, "test_suite_report.html"))
            test_suite_dict = test_suite.as_dict()
        else:
            test_suite_dict = None

        # Đồng bộ Metrics lên MLflow (Đã tích hợp trong Engine)
        self.engine.log_metrics_to_mlflow(
            report_json=report_dict,
            test_suite_json=test_suite_dict
        )

        self.logger.info("✓ Hoàn tất toàn bộ luồng Data Drift Pipeline.")
        return engine_result

if __name__ == "__main__":
    import yaml
    import numpy as np
    
    # 1. Nạp file cấu hình từ YAML
    config_path = "config/pipelines/ml_calories/evidently_profile.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Đã nạp thành công cấu hình từ {config_path}")
    except FileNotFoundError:
        logger.warning(f"Không tìm thấy {config_path}. Đang sử dụng cấu hình mặc định để test.")
        config = {
            "artifact_dir": "Artifacts/reports/ml_calories/data_drift",
            "project_name": "Calories_Prediction_Drift_Test",
            # Giả lập mapping cơ bản nếu không có yaml
            "column_mapping": {
                "numerical_features": ["Age", "Weight", "Duration", "Heart_Rate", "Body_Temp"],
                "categorical_features": ["Gender"],
                "target": "Calories",
                "prediction": "Prediction",
                "task": "regression"
            },
            "drift_options": {"drift_share": 0.5}
        }

    # 2. Tạo hàm phát sinh dữ liệu giả (Mock Data) để chạy thử
    def generate_mock_data(n_samples=200, simulate_drift=False):
        """
        Tạo dữ liệu mô phỏng bài toán Calories Prediction.
        Nếu simulate_drift=True, cố tình làm sai lệch phân phối của Weight và Heart_Rate.
        """
        weight_mean = 85 if simulate_drift else 70  # Tạo drift mạnh ở cân nặng
        hr_mean = 140 if simulate_drift else 110    # Tạo drift ở nhịp tim

        return pd.DataFrame({
            "Age": np.random.randint(18, 60, n_samples),
            "Gender": np.random.choice(["male", "female"], n_samples),
            "Weight": np.random.normal(weight_mean, 10, n_samples),
            "Duration": np.random.normal(30, 10, n_samples),
            "Heart_Rate": np.random.normal(hr_mean, 15, n_samples),
            "Body_Temp": np.random.normal(37.5, 0.5, n_samples),
            "Calories": np.random.normal(300, 50, n_samples),     # Target thực tế
            "Prediction": np.random.normal(310, 55, n_samples)    # Target dự đoán từ mô hình
        })

    logger.info("Đang tạo Reference Data và Current Data (Mock)...")
    df_reference = generate_mock_data(n_samples=500, simulate_drift=False)
    
    # Ở tập hiện tại, ta bật simulate_drift=True để Evidently bắt được Data Drift
    df_current = generate_mock_data(n_samples=200, simulate_drift=True) 

    # 3. (Tùy chọn) Khởi tạo MLflow Tracker
    # Nếu MLOpsTracker chưa setup DB hoàn chỉnh, có thể truyền None để engine bỏ qua ghi log MLflow
    tracker = None 
    
    # Nếu bạn muốn test luôn MLflow, hãy bỏ comment đoạn dưới đây:
    # from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker
    # tracker = MLOpsTracker(experiment_name="Test_Calories_Drift")
    # tracker.start_run("test_evidently_engine_run")

    # 4. Khởi tạo Evaluator và thực thi
    evaluator = CaloriesDriftEvaluator(config=config, mlops_tracker=tracker)
    
    try:
        results = evaluator.execute_evaluation(
            current_df=df_current, 
            reference_df=df_reference
        )
        logger.info(f"✅ Chạy thử nghiệm thành công! Vui lòng mở trình duyệt và xem báo cáo tại:")
        logger.info(f"   -> {os.path.abspath(os.path.join(evaluator.engine.artifact_dir, 'drift_report.html'))}")
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình chạy thử nghiệm: {e}")
        
    # if tracker:
    #     tracker.end_run()
