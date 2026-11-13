from evidently.pipeline.column_mapping import ColumnMapping
import yaml
from typing import Dict, Any

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
