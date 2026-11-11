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

from evidently.metric_preset import DataDriftPreset, DataSummaryPreset
from evidently.metrics import DatasetDriftMetric, DataDriftTable, EmbeddingsDriftMetric, ColumnDriftMetric

from evidently.metric_preset import DataQualityPreset, DataSummaryPreset
from evidently.metrics import (
    DatasetMissingValuesMetric, ColumnMissingValuesMetric, 
    ColumnRegExpMetric, DatasetCorrelationsMetric,
    ColumnValueRangeMetric, ColumnValueListMetric,
    # DataQualityStabilityMetric, ConflictTargetMetric, ConflictPredictionMetric
)

from evidently.metric_preset import ClassificationPreset
from evidently.metrics import (
    ClassificationDummyMetric,
    ClassificationProbDistribution, 
    ClassificationQualityByFeatureTable,
    ClassificationConfusionMatrix
)

from evidently.metric_preset import RegressionPreset
from evidently.metrics import (
    RegressionDummyMetric, 
    RegressionErrorBiasTable, 
    RegressionTopErrorMetric
)

from evidently.metric_preset import TextEvals
from evidently.metrics import (
    ContextQualityLLMEval, FaithfulnessLLMEval, CorrectnessLLMEval, CompletenessLLMEval,
    PIILLMEval, ToxicityLLMEval, DeclineLLMEval, BiasLLMEval, NegativityLLMEval
)
# Các Text Descriptors (Nếu dùng độc lập) có thể import từ evidently.descriptors
from evidently.descriptors import (
    TextLength, WordCount, Sentiment, OOVWordsPercentage, NonLetterCharacterPercentage,
    RegExp, ContainsLink, TextMatch, IsValidJSON, SemanticSimilarity
)

from evidently.metric_preset import RecsysPreset
from evidently.metrics import (
    Personalization, ScoreDistribution, Diversity, Novelty, Serendipity,
    PopularityBiasMetric, ItemBiasMetric, UserBiasMetric, RecCasesTable
)

from evidently.test_suite import TestSuite
from evidently.test_preset import (
    DataDriftTestPreset, 
    DataQualityTestPreset, 
    DataStabilityTestPreset,
    BinaryClassificationTestPreset, 
    MulticlassClassificationTestPreset, 
    RegressionTestPreset, 
    RecsysTestPreset, 
    NoTargetPerformanceTestPreset
)
# Quản lý Workspace
from evidently.ui.workspace import RemoteWorkspace, Workspace
from evidently.ui.dashboards import DashboardPanelPlot, ReportFilter, PanelValue, PlotType


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

        # Tự động tạo thư mục đích nếu chưa tồn tại (vd: Artifacts/reports/ml_calories/evidently)
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
        datetime_cols = column_mapping_cfg.get("datetime_features", [])
        id_col = column_mapping_cfg.get("id_column", None)
        
        # Cấu hình cho ML Task (Classification/Regression) & Embeddings
        target_col = column_mapping_cfg.get("target", None)
        prediction_col = column_mapping_cfg.get("prediction", None)
        task_type = column_mapping_cfg.get("task", None) # "classification" hoặc "regression"
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

        # 2. Cấu hình Mô-đun Dịch chuyển Dữ liệu & Embeddings (Data Drift & Embeddings Drift)
        # Hỗ trợ tự động hoặc cấu hình linh hoạt các StatTests (ks, wasserstein, chisquare, psi...), 
        # Text Drift và Embeddings Drift (Distance, Model, Ratio, MMD).
        
        # --- A. Các tham số Data Drift ---
        drift_share_val = drift_options_cfg.get("dataset_drift_share", None)
        default_stattest = drift_options_cfg.get("default_method", None) 
        default_threshold = drift_options_cfg.get("p_value_alpha", None)
        
        # --- B. Khởi tạo Presets & Metrics Lõi cho Data Drift ---
        drift_preset = DataDriftPreset(
            drift_share=drift_share_val,
            stattest=default_stattest,
            stattest_threshold=default_threshold,
            num_stattest=drift_options_cfg.get("numerical", {}).get("method", None),
            cat_stattest=drift_options_cfg.get("categorical", {}).get("method", None),
            per_column_stattest=drift_options_cfg.get("per_feature_stattest", None)
        )
        
        # Các Metric đơn lẻ (đã được cấu hình mở tham số cho Pipeline)
        dataset_drift_metric = DatasetDriftMetric(drift_share=drift_share_val)
        data_drift_table = DataDriftTable(
            stattest=default_stattest, 
            stattest_threshold=default_threshold
        )
        
        column_drift_metrics = []
        target_columns = drift_options_cfg.get("columns_to_check_drift", [])
        for col in target_columns:
            # Hỗ trợ Text Drift, Categorical, Numerical ở mức cột
            column_drift_metrics.append(ColumnDriftMetric(column_name=col))

        # --- C. Khởi tạo Embeddings Drift Metrics ---
        embeddings_metrics = []
        if embeddings_cfg:
            # Lặp qua các không gian embedding được định nghĩa ở Giai đoạn 1
            for emb_name in embeddings_cfg.keys():
                embeddings_metrics.append(
                    EmbeddingsDriftMetric(
                        embeddings_name=emb_name,
                        drift_method=drift_options_cfg.get("embeddings_drift_method", None) # Hỗ trợ Model, Distance, MMD, Ratio
                    )
                )

        # --- D. Đóng gói Report & Chạy Engine ---
        # Chú ý: Các thành phần độc lập (dataset_drift_metric, data_drift_table) có thể 
        # uncomment và đưa vào list tùy thuộc vào yêu cầu của pipeline.
        active_metrics = [
            DataSummaryPreset(),
            drift_preset,
            # dataset_drift_metric, 
            # data_drift_table
        ] + embeddings_metrics + column_drift_metrics

        report = Report(metrics=active_metrics)
        snapshot = report.run(current_data=curr_dataset, reference_data=ref_dataset)

        # (Tương lai) Dành cho DataDriftTestPreset trong CI/CD Pipelines:
        # from evidently.test_suite import TestSuite
        # from evidently.test_preset import DataDriftTestPreset
        # test_suite = TestSuite(tests=[DataDriftTestPreset(drift_share=drift_share_val)])

     
        # 3. Cấu hình Mô-đun Chất lượng & Tính toàn vẹn Dữ liệu (Data Quality & Integrity)
        # Lấy cấu hình linh hoạt từ file YAML của pipeline (nếu có)
        data_quality_cfg = kwargs.get("data_quality_options", {})
        
        # --- A. Các Presets Lõi cho Data Quality ---
        # Bao phủ tổng quan các thống kê (DatasetStats, ValueStats) và Missing Values
        dq_presets = [
            DataSummaryPreset(),
            DataQualityPreset()
        ]

        # --- B. Tính toàn vẹn dữ liệu (Data Integrity) & RegExp ---
        dq_metrics = []
        
        # Định nghĩa các giá trị được coi là Missing (VD: ["", "n/a", -9999])
        custom_missing_values = data_quality_cfg.get("custom_missing_values", None)
        dq_metrics.append(DatasetMissingValuesMetric(missing_values=custom_missing_values))

        # Phân tích tương quan & Độ ổn định (Correlations & Stability) ở mức độ Dataset
        dq_metrics.append(DatasetCorrelationsMetric())
        
        # Đánh giá xung đột dữ liệu (nếu bài toán có target/prediction)
        if target_col or prediction_col:
            # dq_metrics.append(DataQualityStabilityMetric())
            # dq_metrics.append(ConflictTargetMetric())
            # dq_metrics.append(ConflictPredictionMetric())
            pass

        # --- C. Kiểm tra ở mức độ Từng cột (Column-level Validation) ---
        # Range, List, RegExp và Missing Values per column
        columns_to_check_dq = data_quality_cfg.get("columns_to_check", [])
        
        for col in columns_to_check_dq:
            # 1. Missing values cho từng cột
            dq_metrics.append(ColumnMissingValuesMetric(
                column_name=col, 
                missing_values=custom_missing_values
            ))
            
            # 2. Biểu thức chính quy (RegExp Validation)
            reg_exp_rule = data_quality_cfg.get("regexp_rules", {}).get(col, None)
            if reg_exp_rule:
                dq_metrics.append(ColumnRegExpMetric(
                    column_name=col, 
                    reg_exp=reg_exp_rule
                ))
                
            # 3. Kiểm tra khoảng giá trị (Range Validation)
            range_rules = data_quality_cfg.get("range_rules", {}).get(col, {})
            if range_rules:
                dq_metrics.append(ColumnValueRangeMetric(
                    column_name=col,
                    left=range_rules.get("left", None),
                    right=range_rules.get("right", None)
                ))
                
            # 4. Kiểm tra danh sách giá trị hợp lệ (List Validation)
            list_rules = data_quality_cfg.get("list_rules", {}).get(col, None)
            if list_rules:
                dq_metrics.append(ColumnValueListMetric(
                    column_name=col, 
                    values=list_rules
                ))

        # --- D. Đóng gói vào List Metrics Tổng ---
        # Kết hợp các metrics của Data Quality vào list active_metrics đã có ở phần Drift
        # active_metrics.extend(dq_presets + dq_metrics)
        
        # (Tương lai) Bài kiểm thử tự động (Test Presets) cho CI/CD Pipeline
        # from evidently.test_suite import TestSuite
        # from evidently.test_preset import DataQualityTestPreset, DataStabilityTestPreset
        # dq_test_suite = TestSuite(tests=[
        #     DataQualityTestPreset(),
        #     DataStabilityTestPreset()
        # ])


        # 4. Cấu hình Mô-đun Đánh giá Phân loại (Classification Metrics)
        classification_cfg = kwargs.get("classification_options", {})
        class_metrics = []

        # Chỉ kích hoạt nếu Lược đồ dữ liệu (Giai đoạn 1) định nghĩa đây là bài toán Classification
        if task_type == "classification" and target_col and prediction_col:
            
            # Lấy các tham số cấu hình đặc thù cho Classification
            probas_threshold = classification_cfg.get("probas_threshold", None)
            k_val = classification_cfg.get("k", None)
            
            # --- C. Các Metric Presets Đóng gói sẵn ---
            # Sử dụng ClassificationPreset (Toàn diện nhất) bao gồm:
            # - Chỉ số tổng hợp (Accuracy, Precision, Recall, F1Score, RocAuc, LogLoss...)
            # - Chỉ số nhị phân (TPR, TNR, FPR, FNR)
            # - Chỉ số theo lớp (PrecisionByLabel, RecallByLabel...)
            # - Đồ thị (Confusion Matrix, ROC Curve, PR Curve, Lift Curve...)
            class_metrics.append(
                ClassificationPreset(
                    probas_threshold=probas_threshold, 
                    k=k_val
                )
            )

            # --- D. Mô hình Đối chứng Baseline (Dummy Metrics) ---
            # Bật/tắt so sánh hiệu năng với mô hình đoán ngẫu nhiên/theo tần suất
            if classification_cfg.get("enable_dummy_metrics", False):
                class_metrics.append(ClassificationDummyMetric())

            # --- E. Công cụ Trực quan hóa & Đồ thị Mở rộng (Cấu hình lẻ) ---
            # Nếu người dùng muốn phân tích siêu chi tiết từng thuộc tính và phân bố xác suất:
            if classification_cfg.get("enable_detailed_plots", False):
                class_metrics.append(ClassificationProbDistribution())
                class_metrics.append(ClassificationQualityByFeatureTable())
                # Bảng Confusion Matrix độc lập (nếu không dùng preset)
                # class_metrics.append(ClassificationConfusionMatrix(probas_threshold=probas_threshold, k=k_val))

            # Kết hợp các metrics Phân loại vào list active_metrics đã có
            # active_metrics.extend(class_metrics)
            
            # --- 4. (Tương lai) Bài kiểm thử Tự động (Test Presets) cho CI/CD ---
            # from evidently.test_suite import TestSuite
            # from evidently.test_preset import BinaryClassificationTestPreset, MulticlassClassificationTestPreset
            # class_test_suite = TestSuite(tests=[
            #     BinaryClassificationTestPreset(probas_threshold=probas_threshold)
            #     # Hoặc: MulticlassClassificationTestPreset()
            # ])

        # 5. Cấu hình Mô-đun Đánh giá Hồi quy (Regression Metrics)
        regression_cfg = kwargs.get("regression_options", {})
        reg_metrics = []

        # Chỉ kích hoạt nếu Lược đồ dữ liệu định nghĩa đây là bài toán Regression
        if task_type == "regression" and target_col and prediction_col:
            
            # --- C. Các Metric Presets Đóng gói sẵn ---
            # Sử dụng RegressionPreset (Toàn diện nhất) bao gồm:
            # - Chỉ số tổng hợp (MAE, RMSE, MAPE, R2Score, MeanError, AbsMaxError)
            # - Đồ thị (Predicted vs Actual, Error Plot, Error Distribution, Q-Q Plot...)
            reg_metrics.append(
                RegressionPreset()
            )

            # --- D. Mô hình Đối chứng Baseline (Dummy Metrics) ---
            # So sánh hiệu năng thực tế với mô hình giả định (median/mean baseline)
            if regression_cfg.get("enable_dummy_metrics", False):
                reg_metrics.append(RegressionDummyMetric())

            # --- E. Công cụ Trực quan hóa & Phân tích Độ lệch Sai số (Cấu hình lẻ) ---
            # Bật phân tích lệch (Bias) và Top Error theo từng thuộc tính đầu vào
            if regression_cfg.get("enable_detailed_plots", False):
                columns_to_analyze = regression_cfg.get("columns_to_analyze", None)
                reg_metrics.append(RegressionErrorBiasTable(columns=columns_to_analyze))
                reg_metrics.append(RegressionTopErrorMetric())

            # Cấu hình độc lập lẻ tẻ (nếu không dùng preset tổng)
            # VD: Cấu hình xử lý điểm dữ liệu target=0 cho MAPE
            # zero_handling = regression_cfg.get("zero_handling", None) # "drop", "replace", "none"
            # epsilon = regression_cfg.get("epsilon", None)

            # Kết hợp các metrics Hồi quy vào list active_metrics đã có
            # active_metrics.extend(reg_metrics)
            
            # --- 5. (Tương lai) Bài kiểm thử Tự động (Test Presets) cho CI/CD ---
            # from evidently.test_suite import TestSuite
            # from evidently.test_preset import RegressionTestPreset
            # reg_test_suite = TestSuite(tests=[
            #     RegressionTestPreset()
            # ])

        # 6. Cấu hình Mô-đun Văn bản & LLM Evals (Generative AI / RAG Validation)
        llm_evals_cfg = kwargs.get("llm_evals_options", {})
        text_llm_metrics = []

        # Chỉ kích hoạt nếu có dữ liệu text hoặc pipeline có cấu hình LLM Evals
        if text_cols or llm_evals_cfg:
            
            # --- A. Các Presets Đóng gói sẵn ---
            # TextEvals tính toán phân phối thống kê (TextLength, Sentiment, OOV...) cho toàn bộ text_columns
            if llm_evals_cfg.get("enable_text_evals_preset", False):
                text_llm_metrics.append(TextEvals())

            # --- B. Trích xuất thuộc tính văn bản (Text Descriptors) ---
            # (Thường được sử dụng tích hợp bên trong TextEvals hoặc DataDriftPreset)
            # Tuy nhiên, pipeline có thể định nghĩa ánh xạ đánh giá từng cột độc lập.
            descriptors_cfg = llm_evals_cfg.get("descriptors", {})
            for col, d_list in descriptors_cfg.items():
                # Ví dụ: d_list = ["TextLength", "Sentiment", "OOVWordsPercentage", "ContainsLink"]
                # Logic ánh xạ string -> Descriptor objects sẽ được triển khai tại đây
                pass

            # --- C. Đánh giá RAG & Hệ thống LLM (LLM-as-a-Judge) ---
            llm_judge_cfg = llm_evals_cfg.get("llm_judge", {})
            
            if llm_judge_cfg:
                # LLM Provider (VD: từ module evidently.llm_engine, OpenAI, v.v.)
                # Nhận động từ pipeline để tránh hardcode API Key trong engine
                llm_provider = llm_judge_cfg.get("provider", None) 

                # 1. Bộ ba chỉ số đánh giá RAG (RAG Triad) & Completeness
                if llm_judge_cfg.get("context_quality"):
                    text_llm_metrics.append(ContextQualityLLMEval(provider=llm_provider))
                if llm_judge_cfg.get("faithfulness"):
                    text_llm_metrics.append(FaithfulnessLLMEval(provider=llm_provider))
                if llm_judge_cfg.get("correctness"):
                    text_llm_metrics.append(CorrectnessLLMEval(provider=llm_provider))
                if llm_judge_cfg.get("completeness"):
                    text_llm_metrics.append(CompletenessLLMEval(provider=llm_provider))

                # 2. Đánh giá An toàn & Tuân thủ Policy (Safety Judges)
                if llm_judge_cfg.get("pii"):
                    text_llm_metrics.append(PIILLMEval(provider=llm_provider))
                if llm_judge_cfg.get("toxicity"):
                    text_llm_metrics.append(ToxicityLLMEval(provider=llm_provider))
                if llm_judge_cfg.get("decline"):
                    text_llm_metrics.append(DeclineLLMEval(provider=llm_provider))
                if llm_judge_cfg.get("bias"):
                    text_llm_metrics.append(BiasLLMEval(provider=llm_provider))
                if llm_judge_cfg.get("negativity"):
                    text_llm_metrics.append(NegativityLLMEval(provider=llm_provider))

            # Kết hợp các metrics Text/LLM vào list active_metrics đã có
            # active_metrics.extend(text_llm_metrics)
            
            # --- D. (Tương lai) Các công cụ LLM Datagen & Guardrails ---
            # Lưu ý: Các modules evidently.llm.datagen (tạo dữ liệu giả lập) 
            # và evidently.guardrails (kiểm soát luồng) chạy độc lập với Report, 
            # nên sẽ được cấu hình ở các Gateway/API chặn đầu vào thay vì trong Engine này.

        # 7. Cấu hình Mô-đun Hệ thống Gợi ý & Xếp hạng (Recommendation Systems - RecSys)
        recsys_cfg = kwargs.get("recsys_options", {})
        recsys_metrics = []

        # Kích hoạt nếu Lược đồ dữ liệu định nghĩa bài toán là RecSys (Gợi ý/Xếp hạng)
        # Yêu cầu phải có target_col, prediction_col và tối thiểu user_id, item_id
        if task_type == "recsys" and target_col and prediction_col:
            
            # Lấy tham số K (Top-K) và các cấu hình đặc thù
            k_val = recsys_cfg.get("k", None)
            
            # --- A. Các Metric Presets Đóng gói sẵn ---
            # Sử dụng RecsysPreset (Tự động bao gồm Precision, Recall, MAP, NDCG, MRR, HitRate)
            # Tự động tính Novelty, Diversity, Serendipity, Bias... nếu có truyền đủ train_data và features
            recsys_metrics.append(RecsysPreset(k=k_val))
            
            # --- B. Các Metric Đa dạng, Cá nhân hóa, Độ mới & Thiên vị (Cấu hình lẻ) ---
            # Chỉ sử dụng nếu muốn tinh chỉnh sâu hơn ở bên ngoài Preset
            if recsys_cfg.get("enable_diversity_novelty", False):
                item_features = recsys_cfg.get("item_features", None)
                
                recsys_metrics.append(Personalization(k=k_val))
                recsys_metrics.append(ScoreDistribution(k=k_val))
                recsys_metrics.append(Novelty(k=k_val))
                
                if item_features:
                    recsys_metrics.append(Diversity(k=k_val, item_features=item_features))
                    recsys_metrics.append(Serendipity(k=k_val, item_features=item_features))

            # Đánh giá thiên vị (Bias) theo mức độ phổ biến, thuộc tính User / Item
            if recsys_cfg.get("enable_bias_metrics", False):
                item_bias_cols = recsys_cfg.get("item_bias_columns", None)
                user_bias_cols = recsys_cfg.get("user_bias_columns", None)
                
                recsys_metrics.append(PopularityBiasMetric(k=k_val))
                
                if item_bias_cols:
                    recsys_metrics.append(ItemBiasMetric(k=k_val, column_names=item_bias_cols))
                if user_bias_cols:
                    recsys_metrics.append(UserBiasMetric(k=k_val, column_names=user_bias_cols))

            # --- C. Bảng trực quan hóa các trường hợp gợi ý thực tế (RecCasesTable) ---
            if recsys_cfg.get("enable_rec_cases", False):
                user_ids_to_check = recsys_cfg.get("user_ids", None)
                display_features = recsys_cfg.get("display_features", None)
                
                recsys_metrics.append(
                    RecCasesTable(
                        user_ids=user_ids_to_check, 
                        display_features=display_features
                    )
                )

            # Kết hợp các metrics RecSys vào list active_metrics đã có
            # active_metrics.extend(recsys_metrics)
            
            # --- D. (Tương lai) Bài kiểm thử Tự động (Test Presets) cho CI/CD ---
            # from evidently.test_suite import TestSuite
            # from evidently.test_preset import RecsysTestPreset
            # recsys_test_suite = TestSuite(tests=[
            #     RecsysTestPreset(k=k_val)
            # ])

        # =====================================================================
        # 8. Mô-đun Kiểm thử Tự động (Automated Testing & Test Suites)
        # =====================================================================
        test_suite_cfg = kwargs.get("test_suite_options", {})
        test_suite = None

        if test_suite_cfg:
            logger.info("--- Khởi tạo Test Suite (CI/CD Pass/Fail Validation) ---")
            test_presets = []
            
            # 8.1. Kiểm thử Drift & Chất lượng Dữ liệu
            if test_suite_cfg.get("data_drift"):
                # Có thể custom ngưỡng (ví dụ: is_critical=False để chỉ cảnh báo thay vì Fail)
                test_presets.append(DataDriftTestPreset())
            if test_suite_cfg.get("data_quality"):
                test_presets.append(DataQualityTestPreset())
            if test_suite_cfg.get("data_stability"):
                test_presets.append(DataStabilityTestPreset())
                
            # 8.2. Kiểm thử Hiệu năng Mô hình theo Task
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
            
            # Kiểm thử khi không có nhãn thực tế (Target)
            elif test_suite_cfg.get("no_target_performance"):
                test_presets.append(NoTargetPerformanceTestPreset())

            # 8.3. Thực thi Test Suite
            if test_presets:
                test_suite = TestSuite(tests=test_presets)
                test_suite.run(
                    current_data=curr_dataset, 
                    reference_data=ref_dataset,
                    additional_data=kwargs.get("additional_data", None)
                )
                
                # Phân tích kết quả CI/CD
                test_summary = test_suite.as_dict()
                total_tests = test_summary["summary"]["all_passed"]
                if not total_tests:
                    logger.warning("⚠️ Test Suite chạy xong nhưng có bài kiểm thử bị FAILED hoặc WARNING!")
                else:
                    logger.info("✓ Tất cả các bài kiểm thử đều PASSED!")

        # =====================================================================
        # 9. Quản lý Workspace & Monitoring UI (Đẩy kết quả lên Dashboard)
        # =====================================================================
        if self.evidently_workspace_url:
            logger.info("--- Đồng bộ kết quả lên Evidently Workspace ---")
            try:
                # 1. Khởi tạo Workspace (Remote hoặc Local)
                if self.evidently_workspace_url.startswith("http"):
                    workspace = RemoteWorkspace(self.evidently_workspace_url)
                else:
                    # Tạo hoặc nạp thư mục workspace local (File System)
                    workspace = Workspace.create(self.evidently_workspace_url) 
                
                # 2. Quản lý Dự án (Project Management)
                # Tìm kiếm dự án theo tên (trả về list)
                existing_projects = workspace.search_project(self.project_name)
                
                if not existing_projects:
                    # Khởi tạo dự án mới nếu chưa tồn tại
                    project = workspace.create_project(self.project_name)
                    project.description = f"Automated MLOps tracking for {self.project_name}"
                    project.save() # Lưu metadata của project
                    logger.info(f"Đã tạo mới Project trong Workspace: {self.project_name}")
                else:
                    project = existing_projects[0]
                
                # 3. Lưu trữ & Quản lý Snapshots (Runs)
                # Ghi nhận snapshot (Report) vào dự án
                workspace.add_report(project.id, report)
                
                # Ghi nhận snapshot (TestSuite) vào dự án nếu có chạy kiểm thử
                if test_suite:
                    workspace.add_test_suite(project.id, test_suite)
                    
                logger.info(f"✓ Đã đẩy Snapshots thành công lên Workspace: {self.evidently_workspace_url}")
                
                # (Tùy chọn tương lai) Dọn dẹp Snapshots cũ / Xóa dự án:
                # runs = workspace.list_runs(project.id)
                # if len(runs) > 100:  # Ví dụ: Chỉ giữ lại 100 runs gần nhất
                #     workspace.delete_run(project.id, runs[-1].id) 
                
            except Exception as e:
                logger.error(f"❌ Lỗi khi đồng bộ Workspace: {e}")

        # Trả về cả Report và Test Suite cho Pipeline xử lý (lưu file json/html cục bộ nếu cần)
        return {
            "report": report,
            "test_suite": test_suite
        }



    def log_metrics_to_mlflow(self, report_json: Dict[str, Any], test_suite_json: Optional[Dict[str, Any]] = None) -> None:
        """
        Giai đoạn 3: Đẩy các chỉ số (Drift, Quality, Performance) và kết quả Test (CI/CD) lên MLflow.
        """
        logger.info("--- [GIAO ĐOẠN 3] Đồng bộ Metrics & Test Results lên MLflow Tracker ---")
        
        if not self.tracker:
            logger.warning("MLOpsTracker không khả dụng. Bỏ qua bước log MLflow.")
            return

        raw_metrics = {}

        # 1. Trích xuất Metrics từ Report (Data Drift, Data Quality, ML Performance)
        metrics_results = report_json.get("metrics", [])
        for item in metrics_results:
            metric_type = item.get("metric", "")
            result = item.get("result", {})

            # Trích xuất Data Drift
            if metric_type == "DatasetDriftMetric":
                raw_metrics["dataset_drift_share"] = result.get("drift_share")
                raw_metrics["dataset_drift_flag"] = int(result.get("dataset_drift", False))
                raw_metrics["number_of_drifted_columns"] = result.get("number_of_drifted_columns")
            
            # Trích xuất Data Quality (Missing Values)
            elif metric_type == "DatasetMissingValuesMetric":
                raw_metrics["dataset_missing_share"] = result.get("current", {}).get("share_of_missing_values")

            # Trích xuất Regression Performance
            elif metric_type == "RegressionQualityMetric":
                raw_metrics["regression_rmse"] = result.get("current", {}).get("rmse")
                raw_metrics["regression_mae"] = result.get("current", {}).get("mae")
                raw_metrics["regression_r2_score"] = result.get("current", {}).get("r2_score")

            # Trích xuất Classification Performance
            elif metric_type == "ClassificationQualityMetric":
                raw_metrics["classification_accuracy"] = result.get("current", {}).get("accuracy")
                raw_metrics["classification_f1"] = result.get("current", {}).get("f1")
                raw_metrics["classification_roc_auc"] = result.get("current", {}).get("roc_auc")

        # 2. Trích xuất kết quả từ Test Suite (CI/CD Pass/Fail)
        if test_suite_json:
            summary = test_suite_json.get("summary", {})
            raw_metrics["test_suite_total_tests"] = summary.get("total_tests", 0)
            raw_metrics["test_suite_all_passed"] = int(summary.get("all_passed", False))
            raw_metrics["test_suite_success_count"] = summary.get("success_tests", 0)
            raw_metrics["test_suite_failed_count"] = summary.get("failed_tests", 0)
            raw_metrics["test_suite_warning_count"] = summary.get("warning_tests", 0)

        # 3. Đẩy lên MLflow (Lọc bỏ các giá trị None để tránh lỗi)
        clean_metrics = {k: v for k, v in raw_metrics.items() if v is not None}
        
        if clean_metrics:
            try:
                # Giả định tracker của bạn có hàm log_metrics
                self.tracker.log_metrics(clean_metrics)
                logger.info(f"✓ Đã ghi nhận {len(clean_metrics)} chỉ số lên MLflow.")
            except AttributeError:
                logger.error("Hàm log_metrics không tồn tại trong MLOpsTracker.")
        else:
            logger.info("Không tìm thấy chỉ số cốt lõi nào để log lên MLflow.")
