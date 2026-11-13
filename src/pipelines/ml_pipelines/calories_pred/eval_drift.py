import os
import json
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
        
        # YAML đặt các thiết lập khởi tạo trong engine_config.
        engine_config = self.config.get("engine_config", {})
        self.engine = EvidentlyDriftEngine(
            artifact_dir=engine_config.get(
                "report_dir", "Artifacts/reports/ml_calories/data_drift"
            ),
            project_name=engine_config.get(
                "project_name", "Calories_Prediction_Drift"
            ),
            # None trong YAML có nghĩa là không đồng bộ lên Workspace.
            evidently_workspace_url=engine_config.get("evidently_workspace_url"),
            mlops_tracker=self.tracker
        )

    def _prepare_column_mapping(self) -> Dict[str, Any]:
        """Chuẩn hóa các tên khóa YAML thành mapping mà EvidentlyDriftEngine dùng."""
        mapping_cfg = dict(self.config.get("column_mapping", {}))

        # Engine/DataDefinition dùng datetime_columns; YAML hiện tại khai báo datetime.
        datetime_cols = mapping_cfg.get(
            "datetime_features", mapping_cfg.get("datetime_columns", mapping_cfg.get("datetime"))
        )
        if datetime_cols is not None:
            mapping_cfg["datetime_features"] = (
                [datetime_cols] if isinstance(datetime_cols, str) else list(datetime_cols)
            )

        # DataDefinition trong engine nhận một id_column. Hỗ trợ alias user/item ID
        # khi config không khai báo id_column trực tiếp.
        if not mapping_cfg.get("id_column"):
            mapping_cfg["id_column"] = mapping_cfg.get("user_id") or mapping_cfg.get("item_id") or mapping_cfg.get("id")

        # EvidentlyDriftEngine dùng một trường prediction. Ưu tiên nhãn dự đoán,
        # sau đó mới dùng xác suất nếu cấu hình không có prediction tổng quát.
        if not mapping_cfg.get("prediction"):
            mapping_cfg["prediction"] = (
                mapping_cfg.get("prediction_labels")
                or mapping_cfg.get("prediction_label")
                or mapping_cfg.get("prediction_probas")
                or mapping_cfg.get("prediction_proba")
            )

        # Chuẩn hóa các nhóm cột để DataDefinition luôn nhận danh sách.
        for key in ("numerical_features", "categorical_features", "text_features", "datetime_features"):
            value = mapping_cfg.get(key)
            if isinstance(value, str):
                mapping_cfg[key] = [value]
            elif value is None:
                mapping_cfg[key] = []

        return mapping_cfg

    def _prepare_drift_options(self) -> Dict[str, Any]:
        """
        Chuẩn hóa data_drift trong YAML sang cấu trúc mà EvidentlyDriftEngine
        nhận qua drift_options_cfg.
        """
        drift_cfg = self.config.get("data_drift", self.config.get("drift_options", {}))

        # YAML dùng per_column_options: {column: {stattest, threshold}}.
        # EvidentlyDriftEngine hiện nhận per_feature_stattest dưới dạng
        # {column: stattest}; phần threshold riêng từng cột chưa được engine dùng.
        per_column_options = drift_cfg.get("per_column_options", {})
        per_feature_stattest = drift_cfg.get("per_column_stattest")
        if per_feature_stattest is None and per_column_options:
            per_feature_stattest = {
                column: options["stattest"]
                for column, options in per_column_options.items()
                if isinstance(options, dict) and options.get("stattest") is not None
            }

        return {
            "dataset_drift_share": drift_cfg.get(
                "dataset_drift_share", drift_cfg.get("drift_share", 0.5)
            ),
            "default_method": drift_cfg.get(
                "default_method", drift_cfg.get("stattest")
            ),
            "p_value_alpha": drift_cfg.get(
                "p_value_alpha", drift_cfg.get("threshold")
            ),
            "numerical": {
                "method": drift_cfg.get("num_stattest"),
                "threshold": drift_cfg.get(
                    "num_stattest_threshold", drift_cfg.get("num_threshold")
                ),
            },
            "categorical": {
                "method": drift_cfg.get("cat_stattest"),
                "threshold": drift_cfg.get(
                    "cat_stattest_threshold", drift_cfg.get("cat_threshold")
                ),
            },
            "text": {
                "method": drift_cfg.get("text_stattest"),
                "threshold": drift_cfg.get("text_stattest_threshold", drift_cfg.get("text_threshold")),
            },
            "per_feature_stattest": per_feature_stattest,
            "columns_to_check_drift": drift_cfg.get("columns", []),
        }

    @staticmethod
    def _snapshot_to_dict(snapshot: Any) -> Dict[str, Any]:
        """Chuyển snapshot đã chạy thành dict, ưu tiên các API trên Snapshot."""
        for method_name in ("dict", "dump_dict", "as_dict"):
            method = getattr(snapshot, method_name, None)
            if callable(method):
                value = method()
                if isinstance(value, dict):
                    return value
                if isinstance(value, str):
                    return json.loads(value)
        raise TypeError("Snapshot không cung cấp dict(), dump_dict() hoặc as_dict() hợp lệ.")

    @staticmethod
    def _write_snapshot_serialization(snapshot: Any, path: str) -> None:
        """Lưu serialization đầy đủ của snapshot, không gọi các hàm trên Report."""
        save_method = getattr(snapshot, "save", None)
        if callable(save_method):
            save_method(path)
            return

        dumps_method = getattr(snapshot, "dumps", None)
        if callable(dumps_method):
            with open(path, "w", encoding="utf-8") as file:
                file.write(dumps_method())
            return

        dump_dict_method = getattr(snapshot, "dump_dict", None)
        if callable(dump_dict_method):
            with open(path, "w", encoding="utf-8") as file:
                json.dump(dump_dict_method(), file, ensure_ascii=False, indent=2, default=str)
            return

        raise TypeError("Snapshot không cung cấp save(), dumps() hoặc dump_dict() để serialize.")

    def execute_evaluation(self, current_df: pd.DataFrame, reference_df: pd.DataFrame):
        """
        Thực thi toàn bộ luồng Drift Detection, Export và Monitoring.
        Tất cả artifact của report được tạo từ snapshot của lần chạy này.
        """
        self.logger.info("Bắt đầu quy trình đánh giá Data Drift...")
        
        # Chuẩn hóa mapping YAML (datetime, id, prediction aliases, ...) trước khi
        # chuyển vào Engine để dựng DataDefinition.
        column_mapping_cfg = self._prepare_column_mapping()
        
        drift_options_cfg = self._prepare_drift_options()
        data_quality_options = self.config.get("data_quality_options", {})
        regression_options = self.config.get("regression_options", {})
        test_suite_options = self.config.get("test_suite_options", {})

        engine_result = self.engine.run_drift_detection(
            current_df=current_df,
            reference_df=reference_df,
            column_mapping_cfg=column_mapping_cfg,
            drift_options_cfg=drift_options_cfg,
            data_quality_options=data_quality_options,
            regression_options=regression_options,
            test_suite_options=test_suite_options
        )
        
        snapshot = engine_result.get("snapshot")
        test_suite = engine_result.get("test_suite")
        artifact_dir = self.engine.artifact_dir
        report_dict: Dict[str, Any] = {}

        if snapshot is not None:
            # HTML: kết xuất HTML của snapshot, hoặc dùng API chuỗi HTML.
            html_path = os.path.join(artifact_dir, "drift_report.html")
            save_html = getattr(snapshot, "save_html", None)
            if callable(save_html):
                save_html(html_path)
            else:
                get_html_str = getattr(snapshot, "get_html_str", None)
                if not callable(get_html_str):
                    raise TypeError("Snapshot không hỗ trợ save_html() hoặc get_html_str().")
                with open(html_path, "w", encoding="utf-8") as file:
                    file.write(get_html_str())

            # JSON của kết quả metrics từ chính snapshot.
            json_path = os.path.join(artifact_dir, "drift_report.json")
            save_json = getattr(snapshot, "save_json", None)
            if callable(save_json):
                save_json(json_path)
            else:
                json_method = getattr(snapshot, "json", None)
                if callable(json_method):
                    with open(json_path, "w", encoding="utf-8") as file:
                        file.write(json_method())
                else:
                    with open(json_path, "w", encoding="utf-8") as file:
                        json.dump(self._snapshot_to_dict(snapshot), file, ensure_ascii=False, indent=2, default=str)

            # Serialization snapshot đầy đủ.
            self._write_snapshot_serialization(
                snapshot, os.path.join(artifact_dir, "drift_snapshot.json")
            )

            # Dict export và dữ liệu chuẩn để gửi cho MLflow.
            report_dict = self._snapshot_to_dict(snapshot)

            # CSV là bảng phẳng hóa các metric trong dict snapshot.
            metrics_data = report_dict.get("metrics", [])
            if isinstance(metrics_data, list) and metrics_data:
                pd.json_normalize(metrics_data).to_csv(
                    os.path.join(artifact_dir, "drift_metrics.csv"), index=False
                )
            else:
                self.logger.warning("Snapshot không có danh sách metrics để xuất drift_metrics.csv.")

            self.logger.info(
                f"✓ Đã xuất artifact từ snapshot (HTML, JSON, Dict, CSV, full snapshot) tại {artifact_dir}"
            )
        else:
            self.logger.warning("Engine không trả về snapshot; bỏ qua export report và MLflow report metrics.")

        if test_suite:
            test_suite.save_html(os.path.join(artifact_dir, "test_suite_report.html"))
            test_suite_dict = test_suite.as_dict()
        else:
            test_suite_dict = None

        self.engine.log_metrics_to_mlflow(
            report_json=report_dict,
            test_suite_json=test_suite_dict
        )

        self.logger.info("✓ Hoàn tất toàn bộ luồng Data Drift Pipeline.")
        return engine_result

if __name__ == "__main__":
    import yaml
    import numpy as np
    
    config_path = "config/pipelines/ml_calories/evidently_profile.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Đã nạp thành công cấu hình từ {config_path}")
    except FileNotFoundError:
        logger.warning(f"Không tìm thấy {config_path}. Đang sử dụng cấu hình mặc định để test.")
        config = {
            "engine_config": {
                "report_dir": "Artifacts/reports/ml_calories/data_drift",
                "project_name": "Calories_Prediction_Drift_Test",
                "evidently_workspace_url": None,
            },
            "column_mapping": {
                "numerical_features": ["age", "weight", "duration", "heart_rate", "body_temp"],
                "categorical_features": ["gender"],
                "target": "calories",
                "prediction": "predicted_calories",
                "task": "regression"
            },
            "data_drift": {"dataset_drift_share": 0.5}
        }

    def generate_mock_data(n_samples=200, simulate_drift=False):
        """Tạo dữ liệu mock theo tên cột trong evidently_profile.yaml."""
        weight_mean = 85 if simulate_drift else 70
        hr_mean = 140 if simulate_drift else 110
        return pd.DataFrame({
            "User_ID": np.arange(n_samples),
            "created_at": pd.date_range(datetime.datetime.now(), periods=n_samples, freq="h"),
            "age": np.random.randint(18, 60, n_samples),
            "gender": np.random.choice(["male", "female"], n_samples),
            "weight": np.random.normal(weight_mean, 10, n_samples),
            "height": np.random.normal(170, 10, n_samples),
            "duration": np.random.normal(30, 10, n_samples),
            "heart_rate": np.random.normal(hr_mean, 15, n_samples),
            "body_temp": np.random.normal(37.5, 0.5, n_samples),
            "calories": np.random.normal(300, 50, n_samples),
            "predicted_calories": np.random.normal(310, 55, n_samples)
        })

    logger.info("Đang tạo Reference Data và Current Data (Mock)...")
    df_reference = generate_mock_data(n_samples=500, simulate_drift=False)
    df_current = generate_mock_data(n_samples=200, simulate_drift=True)

    tracker = None
    evaluator = CaloriesDriftEvaluator(config=config, mlops_tracker=tracker)
    try:
        evaluator.execute_evaluation(current_df=df_current, reference_df=df_reference)
        logger.info("✅ Chạy thử nghiệm thành công! Vui lòng xem báo cáo tại:")
        logger.info(f"   -> {os.path.abspath(os.path.join(evaluator.engine.artifact_dir, 'drift_report.html'))}")
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình chạy thử nghiệm: {e}")
