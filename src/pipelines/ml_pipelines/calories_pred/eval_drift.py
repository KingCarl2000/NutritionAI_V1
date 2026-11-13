from evidently.pipeline.column_mapping import ColumnMapping
import yaml
from typing import Dict, Any
from typing import Any, Dict
import pandas as pd
import os

class CaloriesDriftEvaluator:
    def __init__(self, evidently_config_path: str):
        """
        Khởi tạo Evaluator và nạp cấu hình từ yaml.
        """
        with open(evidently_config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # Nhiệm vụ 1: Chuẩn hóa & Khai báo dữ liệu
        self.column_mapping = self._create_data_definition(self.config.get("column_mapping", {}))

    def _create_data_definition(self, mapping_cfg: Dict[str, Any]) -> ColumnMapping:
        """
        Khởi tạo DataDefinition (ColumnMapping) dựa trên cấu hình khai báo.
        Phân tách các nhóm cột chuyên biệt phục vụ Data Drift.
        """
        column_mapping = ColumnMapping()

        # 1. Các thuộc tính số liên tục và phân loại rời rạc (CẬP NHẬT KEY)
        column_mapping.numerical_features = mapping_cfg.get("numerical_features", [])
        column_mapping.categorical_features = mapping_cfg.get("categorical_features", [])

        # 3. Cột thời gian và Định danh (ID) (CẬP NHẬT KEY CHO DATETIME)
        column_mapping.datetime_features = mapping_cfg.get("datetime", None)
        column_mapping.id = mapping_cfg.get("id_column", None)

        # 4. Khai báo bài toán (Regression)
        column_mapping.target = mapping_cfg.get("target", None)
        column_mapping.prediction = mapping_cfg.get("prediction", None)
        column_mapping.task = mapping_cfg.get("task", "regression")

        # Thiết lập pos_label cho bài toán phân loại nhị phân (nếu có)
        if column_mapping.task == "classification" and "pos_label" in mapping_cfg:
            column_mapping.pos_label = mapping_cfg.get("pos_label")

        return column_mapping

    # ... các hàm khởi tạo và _create_data_definition cũ của bạn ...

    def _prepare_drift_options(self) -> Dict[str, Any]:
        """
        Nhiệm vụ 2: Cấu hình Báo cáo DataDriftPreset.
        Trích xuất các tham số kiểm định thống kê và ngưỡng đánh giá linh hoạt từ cấu hình.
        """
        # Lấy block cấu hình drift từ file evidently_profile.yaml
        # Nếu không có, khởi tạo dictionary rỗng để dùng giá trị mặc định của Evidently
        drift_cfg = self.config.get("drift_options", {})
        
        drift_options_cfg = {
            # Ngưỡng tỷ lệ cột bị drift để đánh dấu dataset_drift = True (mặc định 0.5)
            "dataset_drift_share": drift_cfg.get("drift_share", 0.5),
            
            # Thuật toán kiểm định thống kê và ngưỡng áp dụng chung cho tất cả các cột
            "default_method": drift_cfg.get("stattest", None),
            "p_value_alpha": drift_cfg.get("threshold", None),
            
            # Thiết lập thuật toán kiểm định chuyên biệt cho từng nhóm dữ liệu
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
            
            # Dictionary cấu hình thuật toán riêng biệt cho từng cột cụ thể (VD: {"age": "ks"})
            "per_feature_stattest": drift_cfg.get("per_column_stattest", None),
            
            # Danh sách tên các cột cần đánh giá (Nếu rỗng/None, hệ thống tự đánh giá toàn bộ)
            "columns_to_check_drift": drift_cfg.get("columns", [])
        }
        
        return drift_options_cfg

    def execute_drift_report(self, current_df: pd.DataFrame, reference_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Thực thi đánh giá Data Drift thông qua EvidentlyDriftEngine.
        """
        self.logger.info("Bắt đầu cấu hình và thực thi DataDriftPreset...")
        
        # 1. Trích xuất cấu hình Cột (Từ Nhiệm vụ 1)
        column_mapping_cfg = self.config.get("column_mapping", {})
        
        # 2. Chuẩn bị cấu hình Drift Options (Nhiệm vụ 2)
        drift_options = self._prepare_drift_options()
        
        # 3. Chuyển giao cho Engine xử lý (Gọi hàm run_drift_detection đã viết ở engine)
        result = self.engine.run_drift_detection(
            current_df=current_df,
            reference_df=reference_df,
            column_mapping_cfg=column_mapping_cfg,
            drift_options_cfg=drift_options
        )
        
        self.logger.info("Thực thi DataDriftPreset hoàn tất.")
        return result

    # ... các hàm _prepare_drift_options và execute_drift_report cũ ...

    def export_reports(self, engine_result: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """
        Nhiệm vụ 3: Xuất báo cáo Đa định dạng từ Evidently.
        Xử lý đối tượng Report và xuất ra HTML, JSON, Dictionary và Snapshot.
        """
        self.logger.info(f"Bắt đầu xuất báo cáo đa định dạng tại thư mục: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        report = engine_result.get("report")
        if not report:
            self.logger.error("Không tìm thấy đối tượng Report trong kết quả của Engine.")
            return {}

        # 1. Báo cáo HTML Tương tác (HTML Export)
        html_path = os.path.join(output_dir, "drift_report.html")
        report.save_html(html_path)
        self.logger.info(f"✓ Đã lưu HTML Report tại: {html_path}")

        # 2. Dữ liệu Chuẩn hóa JSON (JSON Export)
        json_path = os.path.join(output_dir, "drift_report.json")
        report.save_json(json_path)
        self.logger.info(f"✓ Đã lưu JSON Report tại: {json_path}")

        # 3. Python Dictionary (Dict Export)
        # Sử dụng as_dict() để chuyển đổi báo cáo thành dictionary trên bộ nhớ (RAM)
        # Phục vụ cho việc bóc tách thông số đẩy lên MLflow
        dict_results = report.as_dict()
        self.logger.info("✓ Đã trích xuất thành công đối tượng Python Dictionary.")

        # 5. Mô hình Snapshot Đầy đủ (Full Snapshot Serialization)
        # Lưu trữ đầy đủ trạng thái (metadata, layout, config) để nạp lại vào Workspace sau này
        snapshot_path = os.path.join(output_dir, "drift_snapshot.json")
        report.save(snapshot_path)
        self.logger.info(f"✓ Đã lưu Full Snapshot Serialization tại: {snapshot_path}")

        # 4. Bảng Dữ liệu pandas.DataFrame (Pandas Export)
        # Trích xuất nhanh các chỉ số từ dict_results thành DataFrame để dễ dàng lọc/tính toán
        try:
            metrics_data = dict_results.get("metrics", [])
            if metrics_data:
                df_metrics = pd.DataFrame(metrics_data)
                csv_path = os.path.join(output_dir, "drift_metrics.csv")
                df_metrics.to_csv(csv_path, index=False)
                self.logger.info(f"✓ Đã xuất Pandas DataFrame (CSV) tại: {csv_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Không thể xuất DataFrame từ metrics: {e}")

        return dict_results

# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    # Cấu hình đường dẫn trỏ tới evidently_profile.yaml
    config_path = "config/pipelines/ml_calories/evidently_profile.yaml"
    
    # Khởi tạo
    evaluator = CaloriesDriftEvaluator(config_path)
    
    print("Column Mapping đã được cấu hình thành công:")
    print(f"- Numerical: {evaluator.column_mapping.numerical_features}")
    print(f"- Categorical: {evaluator.column_mapping.categorical_features}")
    print(f"- Task: {evaluator.column_mapping.task}")

    # ... (Khởi tạo Evaluator, nạp DataFrame hiện tại và tham chiếu) ...
        
        # Thực thi Nhiệm vụ 2
    engine_result = evaluator.execute_drift_report(
        current_df=current_df, 
        reference_df=reference_df
    )
        
        # Thực thi Nhiệm vụ 3
        # Đặt đường dẫn lưu báo cáo (thường là thư mục Artifacts của DVC)
    output_directory = "Artifacts/reports/ml_calories/data_drift"
    dict_metrics = evaluator.export_reports(
        engine_result=engine_result, 
        output_dir=output_directory
        )
        
        # Sau đó có thể dùng dict_metrics này truyền vào log_metrics_to_mlflow của Engine
        # evaluator.engine.log_metrics_to_mlflow(report_json=dict_metrics)
