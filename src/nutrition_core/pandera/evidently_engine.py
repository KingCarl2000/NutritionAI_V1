"""
Module: src.nutrition_core.pandera.evidently_engine
Description: Engine tích hợp giữa Pandera Schema Validation, Evidently AI Data Drift Engine,
            DVC Artifact Management và MLflow Logging cho NutritionAI_V1.
"""

import os
import json
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import pandera as pa
from pandera.errors import SchemaError

# Evidently AI Native Imports
from evidently import Report
from evidently.core.datasets import Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.ui.workspace import RemoteWorkspace

# Core Utilities trong NutritionAI_V1
from src.nutrition_core.logging.logger import logger
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker


class EvidentlyDriftEngine:
    """
    Engine thực thi kiểm định trôi lệch dữ liệu (Drift Detection) 
    và đồng bộ với MLOps Infrastructure.
    """

    def __init__(
        self,
        artifact_dir: str = "Artifacts/reports/ml_calories/evidently",
        evidently_workspace_url: Optional[str] = "http://localhost:8000",
        project_name: str = "NutritionAI_MLOps_Pipeline",
        mlops_tracker: Optional[MLOpsTracker] = None,
    ):
        self.artifact_dir = artifact_dir
        self.evidently_workspace_url = evidently_workspace_url
        self.project_name = project_name
        self.tracker = mlops_tracker

        os.makedirs(self.artifact_dir, exist_ok=True)

    def validate_schema(
        self, df: pd.DataFrame, schema_model: pa.DataFrameModel
    ) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        Giai đoạn 1: Kiểm tra Lược đồ Dữ liệu động với Pandera Schema.
        """
        logger.info("--- [GIAO ĐOẠN 1] Thực thi Pandera Schema Validation ---")
        try:
            validated_df = schema_model.validate(df)
            logger.info("✓ Pandera Validation: PASSED!")
            return True, validated_df
        except SchemaError as err:
            logger.error(f"❌ Pandera Validation FAILED: {err}")
            return False, None

    def run_drift_detection(
        self,
        current_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        column_mapping_cfg: Dict[str, Any],
        drift_options_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Giai đoạn 2: Phát hiện Drift bằng Evidently AI dựa trên cấu hình YAML.
        """
        logger.info("--- [GIAO ĐOẠN 2] Thực thi Evidently AI Drift Detection Engine ---")

        # 1. Trích xuất Schema linh hoạt từ configuration
        num_cols = column_mapping_cfg.get("numerical_features", [])
        cat_cols = column_mapping_cfg.get("categorical_features", [])
        text_cols = column_mapping_cfg.get("text_features", [])

        data_def = DataDefinition(
            numerical_columns=num_cols,
            categorical_columns=cat_cols,
            text_columns=text_cols,
        )

        curr_dataset = Dataset.from_pandas(current_df, data_definition=data_def)
        ref_dataset = Dataset.from_pandas(reference_df, data_definition=data_def)

        # 2. Đọc cấu hình ngưỡng từ global.yaml
        drift_preset = DataDriftPreset(
            drift_share=drift_options_cfg.get("dataset_drift_share", 0.5),
            method=drift_options_cfg.get("default_method", "ks"),
            threshold=drift_options_cfg.get("p_value_alpha", 0.05),
            num_method=drift_options_cfg.get("numerical", {}).get("method", "wasserstein"),
            num_threshold=drift_options_cfg.get("numerical", {}).get("wasserstein_distance", 0.1),
            per_column_method=drift_options_cfg.get("per_feature_stattest"),
            per_column_threshold=drift_options_cfg.get("per_feature_threshold"),
        )

        report = Report(metrics=[DataSummaryPreset(), drift_preset])
        snapshot = report.run(current_data=curr_dataset, reference_data=ref_dataset)

        # 3. Xuất tài nguyên Artifacts
        html_path = os.path.join(self.artifact_dir, "drift_report.html")
        json_path = os.path.join(self.artifact_dir, "drift_metrics.json")

        report.save_html(html_path) # type: ignore
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.json()) # type: ignore

        logger.info(f"✓ Báo cáo HTML đã lưu tại: {html_path}")
        logger.info(f"✓ Chỉ số JSON đã lưu tại: {json_path}")

        # 4. Push Snapshot lên Evidently Remote Workspace (Nếu khả dụng)
        if self.evidently_workspace_url:
            try:
                ws = RemoteWorkspace(self.evidently_workspace_url)
                project = ws.get_or_create_project(self.project_name) # type: ignore
                ws.add_run(project.id, snapshot)
                logger.info(f"✓ Đã đồng bộ Snapshot lên Evidently Service UI.")
            except Exception as e:
                logger.warning(f"⚠️ Không thể kết nối Evidently UI Server: {e}")

        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def log_metrics_to_mlflow(self, validation_json: Dict[str, Any]) -> None:
        """
        Giai đoạn 3: Đẩy chỉ số Drift và Artifacts lên MLflow qua MLOpsTracker của hệ thống.
        """
        logger.info("--- [GIAO ĐOẠN 3] Đồng bộ Metrics & Artifacts lên MLflow Tracker ---")
        
        metrics_results = validation_json.get("metrics", [])
        drift_share, number_of_drifted_columns, dataset_drift = None, None, None

        for item in metrics_results:
            result = item.get("result", {})
            if "drift_share" in result and "number_of_drifted_columns" in result:
                drift_share = result.get("drift_share")
                number_of_drifted_columns = result.get("number_of_drifted_columns")
                dataset_drift = int(result.get("dataset_drift", False))
                break

        # Log qua Tracker của hệ thống nếu khả dụng
        if self.tracker and drift_share is not None:
            raw_metrics = {
                "drift_share": drift_share,
                "number_of_drifted_columns": number_of_drifted_columns,
                "dataset_drift_detected": float(dataset_drift) if dataset_drift is not None else None,
            }

            # Ensure all values are typed strictly as float and remove None
            metrics: dict[str, float] = {
                k: float(v) for k, v in raw_metrics.items() if v is not None
            }

            self.tracker.log_params_and_metrics(metrics=metrics)


            html_path = os.path.join(self.artifact_dir, "drift_report.html")
            json_path = os.path.join(self.artifact_dir, "drift_metrics.json")
            
            if os.path.exists(html_path):
                self.tracker.upload_system_logs(html_path)
            if os.path.exists(json_path):
                self.tracker.upload_system_logs(json_path)
                
            logger.info("✓ Đã log thông số Drift thành công lên MLflow.")