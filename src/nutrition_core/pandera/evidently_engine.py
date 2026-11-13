""" 
Module: src.nutrition_core.pandera.evidently_engine 
Description: Engine tích hợp giữa Pandera Schema Validation, Evidently AI Data Drift Engine, DVC Artifact Management và MLflow Logging cho NutritionAI_V1. 
""" 
import json 
import os 
from datetime import datetime
from typing import Any, Dict, Optional, Tuple 
import pandas as pd 
import pandera as pa 
from pandera.errors import SchemaError 

# ------------------------------------------------------------------------- 
# 1. Core Evidently Imports (Top-level & Workspace) 
# ------------------------------------------------------------------------- 
from evidently import DataDefinition, Dataset, Report
from evidently.ui.workspace import RemoteWorkspace, Workspace
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import (
    DashboardPanelPlot,
    bar_plot_panel,
    counter_panel,
    line_plot_panel,
    pie_plot_panel,
    text_panel,
)

# ------------------------------------------------------------------------- 
# 2. Presets (Metrics) 
# ------------------------------------------------------------------------- 
from evidently.presets import (
    ClassificationPreset,
    DataDriftPreset,
    DataSummaryPreset,
    RecsysPreset,
    RegressionPreset,
    TextEvals,
)

# ------------------------------------------------------------------------- 
# 3. Metrics (Tabular, ML & RecSys Metrics) - New API 0.7+
# ------------------------------------------------------------------------- 
from evidently.metrics import (
    # Data Drift
    DriftedColumnsCount,
    EmbeddingsDrift,
    ValueDrift,
    # Data Quality
    DatasetCorrelations,
    DatasetMissingValueCount,
    GroupBy,
    InListValueCount,
    InRangeValueCount,
    MissingValueCount,
    OutListValueCount,
    OutRangeValueCount,
    # Classification & Regression Dummy
    DummyAccuracy,
    DummyF1Score,
    DummyMAE,
    DummyMAPE,
    DummyPrecision,
    DummyRecall,
    DummyRMSE,
    DummyRocAuc,
    # RecSys
    Diversity,
    ItemBias,
    Novelty,
    Personalization,
    PopularityBiasMetric,
    RecCasesTable,
    ScoreDistribution,
    Serendipity,
    UserBias,
)

# ------------------------------------------------------------------------- 
# 4. Text Descriptors & LLM Evaluators 
# ------------------------------------------------------------------------- 
from evidently.descriptors import ( 
    # LLM Evals  
    BiasLLMEval, CompletenessLLMEval, 
    ContextQualityLLMEval, CorrectnessLLMEval, 
    DeclineLLMEval, FaithfulnessLLMEval, 
    NegativityLLMEval, PIILLMEval, ToxicityLLMEval, 
    # Text Descriptors 
    ContainsLink, IsValidJSON, 
    NonLetterCharacterPercentage, 
    OOVWordsPercentage, RegExp, 
    SemanticSimilarity, Sentiment, 
    TextLength, TextMatch, WordCount, 
) 

# ------------------------------------------------------------------------- 
# 5. Legacy Imports (Cho RegExp Metric & Test Suites CI/CD) 
# -------------------------------------------------------------------------
from evidently.legacy.metrics import ColumnRegExpMetric
from evidently.legacy.test_suite import TestSuite
from evidently.legacy.test_preset import (
    BinaryClassificationTestPreset,
    DataDriftTestPreset,
    DataQualityTestPreset,
    DataStabilityTestPreset,
    MulticlassClassificationTestPreset,
    NoTargetPerformanceTestPreset,
    RecsysTestPreset,
    RegressionTestPreset,
)

# ------------------------------------------------------------------------- 
# 6. Core Internal Utilities (NutritionAI_V1) 
# ------------------------------------------------------------------------- 
from src.nutrition_core.logging.logger import logger 
from src.nutrition_core.logging.mlflow_tracker import MLOpsTracker


class EvidentlyDriftEngine:
    """
    Engine thực thi kiểm định trôi lệch dữ liệu (Drift Detection) 
    và đồng bộ với MLOps Infrastructure.
    """
    
    def __init__(
        self,
        artifact_dir: str,          # Bắt buộc truyền từ pipeline
        project_name: str,          # Bắt buộc truyền từ pipeline
        evidently_workspace_url: Optional[str] = None,
        mlops_tracker: Optional[MLOpsTracker] = None,
    ):
        self.artifact_dir = artifact_dir
        self.project_name = project_name
        self.evidently_workspace_url = evidently_workspace_url
        self.tracker = mlops_tracker

        # Tự động tạo thư mục đích nếu chưa tồn tại
        os.makedirs(self.artifact_dir, exist_ok=True)

    def validate_schema(
        self, df: pd.DataFrame, schema_model: pa.DataFrameModel
    ) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        Giai đoạn 1: Kiểm tra Lược đồ Dữ liệu động với Pandera Schema.
        """
        logger.info("--- [GIAI ĐOẠN 1] Thực thi Pandera Schema Validation ---")
        try:
            validated_df = schema_model.validate(df)
            logger.info("✓ Pandera Validation: PASSED!")
            return True, validated_df
        except SchemaError as err:
            logger.error(f"❌ Pandera Validation FAILED: {err}")
            return False, None

    @staticmethod
    def _configure_workspace_dashboard(project: Any, column_mapping_cfg: Dict[str, Any]) -> None:
        """Thêm các panel giám sát mặc định mà không tạo trùng lặp khi chạy lại."""
        dashboard = project.dashboard
        dashboard_model = dashboard.model()
        existing_titles = {panel.title for panel in dashboard_model.panels}

        panels = [
            counter_panel(
                title="Evaluation Row Count",
                description="Số bản ghi trong tập current qua các lần đánh giá.",
                values=[PanelMetric(metric="RowCount", legend="rows")],
                size="half",
                aggregation="last",
            ),
            counter_panel(
                title="Share of Drifted Columns",
                description="Tỷ lệ cột bị đánh dấu drift qua các lần đánh giá.",
                values=[
                    PanelMetric(
                        metric="DriftedColumnsCount",
                        legend="drift share",
                        metric_labels={"value_type": "share"},
                    )
                ],
                size="half",
                aggregation="last",
            ),
        ]

        drift_columns = []
        for key in ("numerical_features", "categorical_features", "text_features"):
            columns = column_mapping_cfg.get(key, []) or []
            if isinstance(columns, str):
                columns = [columns]
            drift_columns.extend(columns)
        for key in ("target", "prediction"):
            column = column_mapping_cfg.get(key)
            if column:
                drift_columns.append(column)

        # Một panel line tổng hợp các cột đã khai báo trong pipeline. Giới hạn danh sách
        # để dashboard không phình to khi schema có nhiều đặc trưng.
        drift_columns = list(dict.fromkeys(drift_columns))[:8]
        if drift_columns:
            panels.append(
                line_plot_panel(
                    title="Feature Drift Over Runs",
                    description="Drift score theo từng lần chạy cho các cột được cấu hình.",
                    values=[
                        PanelMetric(
                            metric="ValueDrift",
                            legend=column,
                            metric_labels={"column": column},
                        )
                        for column in drift_columns
                    ],
                    size="full",
                )
            )

        changed = False
        for panel in panels:
            if panel.title not in existing_titles:
                dashboard.add_panel(panel)
                existing_titles.add(panel.title)
                changed = True

        if changed:
            project.save()

    def run_drift_detection(
        self,
        current_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        column_mapping_cfg: Dict[str, Any],
        drift_options_cfg: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Giai đoạn 2: Phát hiện Drift bằng Evidently AI dựa trên cấu hình YAML.

        Trả về cả Report và Snapshot đã chạy để caller có thể dùng Report cho
        cấu hình hoặc Snapshot nhất quán cho export/serialization.
        """
        logger.info("--- [GIAI ĐOẠN 2] Thực thi Evidently AI Drift Detection Engine ---")

        # 1. Trích xuất Schema linh hoạt từ configuration
        num_cols = column_mapping_cfg.get("numerical_features", [])
        cat_cols = column_mapping_cfg.get("categorical_features", [])
        text_cols = column_mapping_cfg.get("text_features", [])
        datetime_cols = column_mapping_cfg.get("datetime_features", [])
        id_col = column_mapping_cfg.get("id_column", None)
        
        target_col = column_mapping_cfg.get("target", None)
        prediction_col = column_mapping_cfg.get("prediction", None)
        task_type = column_mapping_cfg.get("task", None) 
        pos_label = column_mapping_cfg.get("pos_label", None)
        embeddings_cfg = column_mapping_cfg.get("embeddings", None)

        data_def = DataDefinition(
            numerical_columns=num_cols,
            categorical_columns=cat_cols,
            text_columns=text_cols,
            datetime_columns=datetime_cols,
            id_column=id_col,
            target=target_col,
            prediction=prediction_col,
            task=task_type,
            pos_label=pos_label,
            embeddings=embeddings_cfg
        )

        curr_dataset = Dataset.from_pandas(current_df, data_definition=data_def)
        ref_dataset = Dataset.from_pandas(reference_df, data_definition=data_def)

        # 2. Cấu hình Mô-đun Dịch chuyển Dữ liệu & Embeddings
        drift_share_val = drift_options_cfg.get("dataset_drift_share", None)
        default_stattest = drift_options_cfg.get("default_method", None) 
        default_threshold = drift_options_cfg.get("p_value_alpha", None)
        
        drift_preset = DataDriftPreset(
            drift_share=drift_share_val,
            stattest=default_stattest,
            stattest_threshold=default_threshold,
            num_stattest=drift_options_cfg.get("numerical", {}).get("method", None),
            cat_stattest=drift_options_cfg.get("categorical", {}).get("method", None),
            per_column_stattest=drift_options_cfg.get("per_feature_stattest", None)
        )
        
        # Các Metric đơn lẻ 
        dataset_drift_metric = DriftedColumnsCount(drift_share=drift_share_val)
        
        column_drift_metrics = []
        target_columns = drift_options_cfg.get("columns_to_check_drift", [])
        for col in target_columns:
            column_drift_metrics.append(ValueDrift(column_name=col))

        embeddings_metrics = []
        if embeddings_cfg:
            for emb_name in embeddings_cfg.keys():
                embeddings_metrics.append(
                    EmbeddingsDrift(
                        embeddings_name=emb_name,
                        drift_method=drift_options_cfg.get("embeddings_drift_method", None)
                    )
                )

        active_metrics = [
            DataSummaryPreset(),
            drift_preset,
        ] + embeddings_metrics + column_drift_metrics

        # 3. Cấu hình Mô-đun Chất lượng & Tính toàn vẹn Dữ liệu
        data_quality_cfg = kwargs.get("data_quality_options", {})
        
        dq_metrics = []
        custom_missing_values = data_quality_cfg.get("custom_missing_values", None)
        dq_metrics.append(DatasetMissingValueCount(missing_values=custom_missing_values))
        dq_metrics.append(DatasetCorrelations())
        
        columns_to_check_dq = data_quality_cfg.get("columns_to_check", [])
        for col in columns_to_check_dq:
            dq_metrics.append(MissingValueCount(
                column_name=col, 
                missing_values=custom_missing_values
            ))
            
            reg_exp_rule = data_quality_cfg.get("regexp_rules", {}).get(col, None)
            if reg_exp_rule:
                dq_metrics.append(ColumnRegExpMetric(column_name=col, reg_exp=reg_exp_rule))
                
            range_rules = data_quality_cfg.get("range_rules", {}).get(col, {})
            if range_rules:
                dq_metrics.append(OutRangeValueCount(
                    column_name=col,
                    left=range_rules.get("left", None),
                    right=range_rules.get("right", None)
                ))
                
            list_rules = data_quality_cfg.get("list_rules", {}).get(col, None)
            if list_rules:
                dq_metrics.append(OutListValueCount(column_name=col, values=list_rules))

        active_metrics.extend(dq_metrics)

        # 4. Cấu hình Mô-đun Đánh giá Phân loại
        classification_cfg = kwargs.get("classification_options", {})
        class_metrics = []

        if task_type == "classification" and target_col and prediction_col:
            probas_threshold = classification_cfg.get("probas_threshold", None)
            k_val = classification_cfg.get("k", None)
            
            class_metrics.append(ClassificationPreset(probas_threshold=probas_threshold, k=k_val))

            if classification_cfg.get("enable_dummy_metrics", False):
                class_metrics.extend([
                    DummyAccuracy(), DummyF1Score(), DummyPrecision(), DummyRecall(), DummyRocAuc()
                ])

            active_metrics.extend(class_metrics)

        # 5. Cấu hình Mô-đun Đánh giá Hồi quy
        regression_cfg = kwargs.get("regression_options", {})
        reg_metrics = []

        if task_type == "regression" and target_col and prediction_col:
            reg_metrics.append(RegressionPreset())

            if regression_cfg.get("enable_dummy_metrics", False):
                reg_metrics.extend([DummyMAE(), DummyMAPE(), DummyRMSE()])
                
            active_metrics.extend(reg_metrics)

        # 6. Cấu hình Mô-đun Văn bản & LLM Evals
        llm_evals_cfg = kwargs.get("llm_evals_options", {})
        text_llm_metrics = []

        if text_cols or llm_evals_cfg:
            if llm_evals_cfg.get("enable_text_evals_preset", False):
                text_llm_metrics.append(TextEvals())

            llm_judge_cfg = llm_evals_cfg.get("llm_judge", {})
            if llm_judge_cfg:
                llm_provider = llm_judge_cfg.get("provider", "openai") 
                eval_col = llm_judge_cfg.get("column_name", prediction_col)
                if not eval_col and text_cols:
                    eval_col = text_cols[0]

                if not eval_col:
                    logger.warning("⚠️ Không xác định được cột văn bản (column_name) để chạy LLM Evals.")
                else:
                    if llm_judge_cfg.get("context_quality"):
                        text_llm_metrics.append(ContextQualityLLMEval(column_name=eval_col, provider=llm_provider))
                    if llm_judge_cfg.get("faithfulness"):
                        text_llm_metrics.append(FaithfulnessLLMEval(column_name=eval_col, provider=llm_provider))
                    if llm_judge_cfg.get("correctness"):
                        text_llm_metrics.append(CorrectnessLLMEval(column_name=eval_col, provider=llm_provider))
                    if llm_judge_cfg.get("completeness"):
                        text_llm_metrics.append(CompletenessLLMEval(column_name=eval_col, provider=llm_provider))
                    if llm_judge_cfg.get("pii"):
                        text_llm_metrics.append(PIILLMEval(column_name=eval_col, provider=llm_provider))
                    if llm_judge_cfg.get("toxicity"):
                        text_llm_metrics.append(ToxicityLLMEval(column_name=eval_col, provider=llm_provider))
                    if llm_judge_cfg.get("decline"):
                        text_llm_metrics.append(DeclineLLMEval(column_name=eval_col, provider=llm_provider))
                    if llm_judge_cfg.get("bias"):
                        text_llm_metrics.append(BiasLLMEval(column_name=eval_col, provider=llm_provider))
                    if llm_judge_cfg.get("negativity"):
                        text_llm_metrics.append(NegativityLLMEval(column_name=eval_col, provider=llm_provider))

            active_metrics.extend(text_llm_metrics)

        # 7. Cấu hình Mô-đun Hệ thống Gợi ý (RecSys)
        recsys_cfg = kwargs.get("recsys_options", {})
        recsys_metrics = []

        if task_type == "recsys" and target_col and prediction_col:
            k_val = recsys_cfg.get("k", None)
            recsys_metrics.append(RecsysPreset(k=k_val))
            
            if recsys_cfg.get("enable_diversity_novelty", False):
                item_features = recsys_cfg.get("item_features", None)
                recsys_metrics.append(Personalization(k=k_val))
                recsys_metrics.append(ScoreDistribution(k=k_val))
                recsys_metrics.append(Novelty(k=k_val))
                if item_features:
                    recsys_metrics.append(Diversity(k=k_val, item_features=item_features))
                    recsys_metrics.append(Serendipity(k=k_val, item_features=item_features))

            if recsys_cfg.get("enable_bias_metrics", False):
                item_bias_cols = recsys_cfg.get("item_bias_columns", None)
                user_bias_cols = recsys_cfg.get("user_bias_columns", None)
                recsys_metrics.append(PopularityBiasMetric(k=k_val))
                if item_bias_cols:
                    recsys_metrics.append(ItemBias(k=k_val, column_names=item_bias_cols))
                if user_bias_cols:
                    recsys_metrics.append(UserBias(k=k_val, column_names=user_bias_cols))

            if recsys_cfg.get("enable_rec_cases", False):
                user_ids_to_check = recsys_cfg.get("user_ids", None)
                display_features = recsys_cfg.get("display_features", None)
                recsys_metrics.append(
                    RecCasesTable(user_ids=user_ids_to_check, display_features=display_features)
                )

            active_metrics.extend(recsys_metrics)

        # Chạy Report tại một timestamp duy nhất và sử dụng Snapshot này cho Workspace/export.
        report = Report(metrics=active_metrics)
        run_timestamp = kwargs.get("timestamp") or datetime.now()
        snapshot = report.run(
            current_data=curr_dataset,
            reference_data=ref_dataset,
            timestamp=run_timestamp,
        )

        # =====================================================================
        # 8. Mô-đun Kiểm thử Tự động (Legacy Test Suites cho CI/CD)
        # =====================================================================
        test_suite_cfg = kwargs.get("test_suite_options", {})
        test_suite = None

        if test_suite_cfg:
            logger.info("--- Khởi tạo Test Suite (CI/CD Pass/Fail Validation) ---")
            test_presets = []
            
            if test_suite_cfg.get("data_drift"):
                test_presets.append(DataDriftTestPreset())
            if test_suite_cfg.get("data_quality"):
                test_presets.append(DataQualityTestPreset())
            if test_suite_cfg.get("data_stability"):
                test_presets.append(DataStabilityTestPreset())
                
            if task_type == "classification" and target_col:
                if test_suite_cfg.get("binary_classification"):
                    test_presets.append(BinaryClassificationTestPreset())
                elif test_suite_cfg.get("multiclass_classification"):
                    test_presets.append(MulticlassClassificationTestPreset())
            
            elif task_type == "regression" and target_col:
                if test_suite_cfg.get("regression"):
                    test_presets.append(RegressionTestPreset())
                    
            elif task_type == "recsys" and target_col:
                if test_suite_cfg.get("recsys"):
                    test_presets.append(RecsysTestPreset(k=test_suite_cfg.get("k", 10)))
            
            elif test_suite_cfg.get("no_target_performance"):
                test_presets.append(NoTargetPerformanceTestPreset())

            if test_presets:
                test_suite = TestSuite(tests=test_presets)
                test_suite.run(
                    current_data=curr_dataset, 
                    reference_data=ref_dataset,
                    additional_data=kwargs.get("additional_data", None)
                )
                
                test_summary = test_suite.as_dict()
                total_tests = test_summary["summary"]["all_passed"]
                if not total_tests:
                    logger.warning("⚠️ Test Suite chạy xong nhưng có bài kiểm thử bị FAILED hoặc WARNING!")
                else:
                    logger.info("✓ Tất cả các bài kiểm thử đều PASSED!")

        # =====================================================================
        # 9. Đồng bộ Snapshot và Dashboard lên Evidently Workspace
        # =====================================================================
        if self.evidently_workspace_url:
            logger.info("--- Đồng bộ Snapshot lên Evidently Workspace ---")
            try:
                if self.evidently_workspace_url.startswith("http"):
                    workspace = RemoteWorkspace(self.evidently_workspace_url)
                else:
                    workspace = Workspace.create(self.evidently_workspace_url)
                
                existing_projects = workspace.search_project(self.project_name)
                if existing_projects:
                    project = existing_projects[0]
                    project.description = f"Automated MLOps tracking for {self.project_name}"
                    project.save()
                else:
                    project = workspace.create_project(
                        self.project_name,
                        description=f"Automated MLOps tracking for {self.project_name}",
                    )
                    logger.info(f"Đã tạo mới Project trong Workspace: {self.project_name}")

                self._configure_workspace_dashboard(project, column_mapping_cfg)
                workspace.add_run(project.id, snapshot)

                if test_suite:
                    workspace.add_test_suite(project.id, test_suite)

                logger.info(
                    f"✓ Đã thêm Snapshot (timestamp={run_timestamp.isoformat()}) vào project "
                    f"{self.project_name} tại Workspace: {self.evidently_workspace_url}"
                )
                
            except Exception as e:
                logger.error(f"❌ Lỗi khi đồng bộ Workspace: {e}")

        return {
            "report": report,
            "snapshot": snapshot,
            "test_suite": test_suite
        }

    def log_metrics_to_mlflow(self, report_json: Dict[str, Any], test_suite_json: Optional[Dict[str, Any]] = None) -> None:
        """
        Giai đoạn 3: Đẩy các chỉ số (Drift, Quality, Performance) và kết quả Test (CI/CD) lên MLflow.
        """
        logger.info("--- [GIAI ĐOẠN 3] Đồng bộ Metrics & Test Results lên MLflow Tracker ---")
        
        if not self.tracker:
            logger.warning("MLOpsTracker không khả dụng. Bỏ qua bước log MLflow.")
            return

        raw_metrics = {}

        metrics_results = report_json.get("metrics", [])
        for item in metrics_results:
            metric_type = item.get("metric", "")
            result = item.get("result", {})

            # Cập nhật tên metric theo bản mới 0.7.23
            if metric_type == "DriftedColumnsCount":
                raw_metrics["dataset_drift_share"] = result.get("drift_share")
                raw_metrics["dataset_drift_flag"] = int(result.get("dataset_drift", False))
                raw_metrics["number_of_drifted_columns"] = result.get("number_of_drifted_columns")
            
            elif metric_type == "DatasetMissingValueCount":
                raw_metrics["dataset_missing_share"] = result.get("current", {}).get("share_of_missing_values")

            elif metric_type == "RegressionQualityMetric":
                raw_metrics["regression_rmse"] = result.get("current", {}).get("rmse")
                raw_metrics["regression_mae"] = result.get("current", {}).get("mae")
                raw_metrics["regression_r2_score"] = result.get("current", {}).get("r2_score")

            elif metric_type == "ClassificationQualityMetric":
                raw_metrics["classification_accuracy"] = result.get("current", {}).get("accuracy")
                raw_metrics["classification_f1"] = result.get("current", {}).get("f1")
                raw_metrics["classification_roc_auc"] = result.get("current", {}).get("roc_auc")

        if test_suite_json:
            summary = test_suite_json.get("summary", {})
            raw_metrics["test_suite_total_tests"] = summary.get("total_tests", 0)
            raw_metrics["test_suite_all_passed"] = int(summary.get("all_passed", False))
            raw_metrics["test_suite_success_count"] = summary.get("success_tests", 0)
            raw_metrics["test_suite_failed_count"] = summary.get("failed_tests", 0)
            raw_metrics["test_suite_warning_count"] = summary.get("warning_tests", 0)

        clean_metrics = {k: v for k, v in raw_metrics.items() if v is not None}
        
        if clean_metrics:
            try:
                self.tracker.log_metrics(clean_metrics)
                logger.info(f"✓ Đã ghi nhận {len(clean_metrics)} chỉ số lên MLflow.")
            except AttributeError:
                logger.error("Hàm log_metrics không tồn tại trong MLOpsTracker.")
        else:
            logger.info("Không tìm thấy chỉ số cốt lõi nào để log lên MLflow.")

#**Một số điểm lưu ý sau khi nâng cấp:**
#* File này đã giải quyết hoàn toàn lỗi `ImportError` ở môi trường ảo.
#* Biến `kwargs` đã được bổ sung vào khai báo hàm `run_drift_detection(...)` để nó có thể nhận cấu hình bổ sung (`data_quality_options`, `regression_options`, v.v.) mà caller cần truyền.
#* Dữ liệu bóc tách cho MLflow (như `DriftedColumnsCount`) đã được đổi chuỗi để đảm bảo JSON parser có thể nhận diện chính xác các metric được sinh ra từ Evidently.
