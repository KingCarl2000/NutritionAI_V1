
import importlib
import yaml
from pathlib import Path
from typing import Type, Any, Dict

def load_class_from_registry(api_path: str) -> Any:
    """
    Nạp động Class, Function hoặc Method từ đường dẫn dot-notation trong YAML.
    Hỗ trợ cả 2 dạng:
    - 'package.module.ClassName'
    - 'package.module.ClassName.method_name'
    """
    if not api_path or not isinstance(api_path, str):
        raise ValueError("`api_path` phải là chuỗi ký tự hợp lệ.")

    parts = api_path.split(".")
    
    # Thử import module từ dài nhất đến ngắn nhất
    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        try:
            obj = importlib.import_module(module_path)
            # Duyệt qua các attr còn lại (Class -> Method/Property)
            for attr in parts[i:]:
                obj = getattr(obj, attr)
            return obj
        except ImportError:
            continue
            
    raise ImportError(f"Không thể import hoặc truy cập thành phần từ đường dẫn API: '{api_path}'")


class DataValidationConfig:
    """
    Lớp quản lý và khởi tạo cấu hình cho quy trình Data Validation từ các file YAML.
    """
    def __init__(
        self, 
        global_config_path: str = "config/pipelines/global.yaml",
        pipeline_config_path: str = "config/pipelines/ml_calories/data_val.yaml",
        evidently_config_path: str = "config/pipelines/ml_calories/evidently_profile.yaml"
    ):
        self.base_dir = Path(__file__).resolve().parents[4] # Điều chỉnh tùy thuộc vào vị trí chạy lệnh
        
        # Load cấu hình từ các file YAML
        self.global_config = self._load_yaml(global_config_path)
        self.pipeline_config = self._load_yaml(pipeline_config_path)
        self.evidently_config = self._load_yaml(evidently_config_path)

        # Trích xuất các tham số quan trọng
        self._extract_configs()

    def _load_yaml(self, rel_path: str) -> Dict[str, Any]:
        """Đọc và parse file YAML."""
        file_path = self.base_dir / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _extract_configs(self):
        """Khởi tạo các tham số định tuyến và pipeline specific."""
        # 1. Đường dẫn thư mục DVC
        dvc_paths = self.pipeline_config.get("pipeline_paths", {}).get("dvc_tracked_dirs", {})
        self.input_dir = self.base_dir / dvc_paths.get("input_dir")
        self.output_dir = self.base_dir / dvc_paths.get("output_dir")
        self.quarantine_dir = self.base_dir / dvc_paths.get("quarantine_dir")
        
        # 2. Lấy API paths của Pandera Schema
        schema_classes = self.pipeline_config.get("schema_classes", {})
        self._train_schema_path = schema_classes.get("train_schema")
        self._inference_schema_path = schema_classes.get("inference_schema")
        
        # 3. MLOps Thresholds (từ global.yaml)
        failure_rates = self.global_config.get("mlops_thresholds", {}).get("data_failure_rate", {})
        self.info_threshold = failure_rates.get("info_threshold")
        self.warning_threshold = failure_rates.get("warning_threshold")

    def get_schema(self, mode: str = "train") -> Type[Any]:
        """
        Tải và trả về lớp Pandera Schema dựa trên chế độ thực thi.
        
        Args:
            mode (str): 'train' cho huấn luyện, 'inference' cho dự đoán.
        """
        if mode == "train":
            if not self._train_schema_path:
                raise ValueError("Không tìm thấy cấu hình 'train_schema' trong data_val.yaml")
            return load_class_from_registry(self._train_schema_path)
        elif mode == "inference":
            if not self._inference_schema_path:
                raise ValueError("Không tìm thấy cấu hình 'inference_schema' trong data_val.yaml")
            return load_class_from_registry(self._inference_schema_path)
        else:
            raise ValueError("Chế độ (mode) chỉ hỗ trợ 'train' hoặc 'inference'.")

# Test nhanh (Tùy chọn - Dùng để kiểm tra khi chạy trực tiếp file)
if __name__ == "__main__":
    try:
        config_manager = DataValidationConfig()
        
        # Nạp schema huấn luyện
        TrainSchema = config_manager.get_schema(mode="train")
        print(f"Đã nạp thành công Schema Huấn luyện: {TrainSchema.__name__}")
        
        print(f"Thư mục đầu ra (Validated): {config_manager.output_dir}")
        print(f"Ngưỡng Fail-Fast (Global): {config_manager.warning_threshold * 100}%")
        
    except Exception as e:
        print(f"Lỗi khởi tạo cấu hình: {e}")