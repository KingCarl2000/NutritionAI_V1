import os
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.data
from typing import Dict, Any, Optional, List

# Kế thừa logger hiện tại và các đường dẫn file log từ logger.py
from src.nutrition_core.logging.logger import logger, OPER_LOG_FILE_PATH, ERROR_LOG_FILE_PATH

# Xác định đường dẫn gốc dự án (Tương tự cách làm trong logger.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REQUIREMENTS_DIR = os.path.join(PROJECT_ROOT, "requirements")

class MLOpsTracker:
    """
    Lớp quản lý MLflow Tracking, tích hợp đọc cấu hình môi trường từ thư mục requirements/.
    """
    
    def __init__(self, experiment_name: str, tracking_uri: str = "sqlite:///mlflow.db"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        try:
            os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
            mlflow.enable_system_metrics_logging()
            logger.info("Đã bật tự động theo dõi tài nguyên phần cứng (CPU, GPU, RAM) bằng MLflow.")
        except Exception as e:
            logger.warning(f"Không thể bật System Metrics Logging: {e}")

    def start_run(self, run_name: Optional[str] = None):
        return mlflow.start_run(run_name=run_name)

    def upload_system_logs(self):
        try:
            if os.path.exists(OPER_LOG_FILE_PATH):
                mlflow.log_artifact(OPER_LOG_FILE_PATH, artifact_path="system_logs")
            if os.path.exists(ERROR_LOG_FILE_PATH):
                mlflow.log_artifact(ERROR_LOG_FILE_PATH, artifact_path="system_logs")
            logger.info("Đã đồng bộ file log hệ thống lên MLflow Artifacts.")
        except Exception as e:
            logger.error(f"Lỗi khi upload log lên MLflow: {e}")

    def log_params_and_metrics(self, params: Dict[str, Any] = None, metrics: Dict[str, float] = None, step: int = None):
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics, step=step)

    def log_error_and_tag(self, error_msg: str, tags: Dict[str, str] = None):
        logger.error(error_msg)
        if tags:
            mlflow.set_tags(tags)
        mlflow.set_tag("status", "FAILED")
        mlflow.set_tag("error_message", error_msg[:250]) 

    def log_dataset_lineage(self, df: pd.DataFrame, dataset_name: str, source_uri: str):
        try:
            dataset = mlflow.data.from_pandas(df, name=dataset_name, source=source_uri)
            mlflow.log_input(dataset, context="training")
        except Exception as e:
            logger.error(f"Không thể log dataset lineage: {e}")

    # =========================================================================
    # CÁC HÀM XỬ LÝ MÔI TRƯỜNG & ĐÓNG GÓI MÔ HÌNH (CẬP NHẬT THEO PROJECT TREE)
    # =========================================================================
    
    def _get_pip_requirements(self, framework: str) -> List[str]:
        """
        Đọc và gộp các file requirements tương ứng để đóng gói cùng mô hình.
        Luôn tải base.txt, sau đó tải thêm file chuyên biệt (ml.txt, dl.txt, llm.txt).
        """
        requirements = []
        
        # 1. Luôn đọc base.txt (Chứa fastapi, pydantic, psycopg, dvc...)
        base_req_path = os.path.join(REQUIREMENTS_DIR, "base.txt")
        if os.path.exists(base_req_path):
            with open(base_req_path, "r", encoding="utf-8") as f:
                requirements.extend([line.strip() for line in f if line.strip() and not line.startswith("#")])
        else:
            logger.warning(f"Không tìm thấy file base requirements tại: {base_req_path}")

        # 2. Xác định file requirements mở rộng theo framework
        specific_file = None
        if framework in ["sklearn", "xgboost"]:
            specific_file = "ml.txt"
        elif framework in ["pytorch", "tensorflow"]:
            specific_file = "dl.txt"
        elif framework in ["openai", "langchain"]:
            specific_file = "llm.txt"

        # 3. Đọc và gộp file mở rộng
        if specific_file:
            specific_req_path = os.path.join(REQUIREMENTS_DIR, specific_file)
            if os.path.exists(specific_req_path):
                with open(specific_req_path, "r", encoding="utf-8") as f:
                    requirements.extend([line.strip() for line in f if line.strip() and not line.startswith("#")])
            else:
                logger.warning(f"Không tìm thấy requirements mở rộng tại: {specific_req_path}")
                
        return requirements

    def log_model(self, model, artifact_path: str = "model", framework: str = "sklearn"):
        """
        Đóng gói mô hình và tự động nhúng danh sách thư viện từ thư mục requirements/.
        """
        # Lấy danh sách dependencies động dựa vào framework đang chạy
        pip_reqs = self._get_pip_requirements(framework)
        
        try:
            if framework == "sklearn":
                import mlflow.sklearn
                mlflow.sklearn.log_model(model, artifact_path, pip_requirements=pip_reqs)
            elif framework == "xgboost":
                import mlflow.xgboost
                mlflow.xgboost.log_model(model, artifact_path, pip_requirements=pip_reqs)
            elif framework == "pytorch":
                import mlflow.pytorch
                mlflow.pytorch.log_model(model, artifact_path, pip_requirements=pip_reqs)
            
            logger.info(f"Đã đóng gói mô hình vào '{artifact_path}' kèm môi trường chuẩn từ requirements ({len(pip_reqs)} packages).")
        except Exception as e:
            logger.error(f"Lỗi khi đóng gói mô hình: {e}")

"""
    @staticmethod
    def enable_llm_tracing(framework: str = "openai"):
        try:
            if framework == "openai":
                mlflow.openai.autolog()
            elif framework == "langchain":
                mlflow.langchain.autolog()
            logger.info(f"Đã bật MLflow Tracing cho thư viện: {framework}")
        except Exception as e:
            logger.error(f"Lỗi khi bật LLM Tracing: {e}")
"""