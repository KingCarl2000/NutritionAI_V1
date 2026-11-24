import inspect
import json
import os
from pathlib import Path
import pandas as pd
import mlflow
from mlflow import data, xgboost, pytorch, sklearn
from typing import Dict, Any, Optional, List

# Kế thừa logger hiện tại và các đường dẫn file log từ logger.py
from src.nutrition_core.logging.logger import (
    logger,
    OPER_LOG_FILE_PATH,
    ERROR_LOG_FILE_PATH,
)

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
            logger.info(
                "Đã bật tự động theo dõi tài nguyên phần cứng (CPU, GPU, RAM) bằng MLflow."
            )
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

    def log_params_and_metrics(
        self,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        step: Optional[int] = None,
    ):
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics, step=step)

    def log_error_and_tag(self, error_msg: str, tags: Optional[Dict[str, str]] = None):
        logger.error(error_msg)
        if tags:
            mlflow.set_tags(tags)
        mlflow.set_tag("status", "FAILED")
        mlflow.set_tag("error_message", error_msg[:250])

    def log_dataset_lineage(self, df: pd.DataFrame, dataset_name: str, source_uri: str):
        try:
            from_pandas = getattr(data, "from_pandas")
            dataset = from_pandas(df, name=dataset_name, source=source_uri)
            mlflow.log_input(dataset, context="training")
        except Exception as e:
            logger.error(f"Không thể log dataset lineage: {e}")

    # =========================================================================
    # CÁC HÀM XỬ LÝ MÔI TRƯỜNG & ĐÓNG GÓI MÔ HÌNH (CẬP NHẬT THEO PROJECT TREE)
    # =========================================================================

    def _read_requirements_file(
        self, requirements_path: str, visited: Optional[set[str]] = None
    ) -> List[str]:
        path = Path(requirements_path).resolve()
        visited = visited or set()
        if str(path) in visited:
            return []
        visited.add(str(path))

        requirements: List[str] = []
        if not path.is_file():
            logger.warning(f"Không tìm thấy file requirements tại: {path}")
            return requirements

        with path.open("r", encoding="utf-8") as requirements_file:
            for raw_line in requirements_file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(("-r ", "--requirement ")):
                    included_path = line.split(maxsplit=1)[1]
                    requirements.extend(
                        self._read_requirements_file(
                            str(path.parent / included_path), visited
                        )
                    )
                elif line.startswith(("-e ", "--editable ")):
                    continue
                else:
                    requirements.append(line)
        return requirements

    def _get_pip_requirements(self, framework: str) -> List[str]:
        """
        Đọc và gộp các file requirements tương ứng để đóng gói cùng mô hình.
        Luôn tải base.txt, sau đó tải thêm file chuyên biệt (ml.txt, dl.txt, llm.txt).
        """
        base_req_path = os.path.join(REQUIREMENTS_DIR, "base.txt")
        requirements = self._read_requirements_file(base_req_path)

        specific_file = None
        if framework in ["sklearn", "xgboost"]:
            specific_file = "ml.txt"
        elif framework in ["pytorch", "tensorflow"]:
            specific_file = "dl.txt"
        elif framework in ["openai", "langchain"]:
            specific_file = "llm.txt"

        if specific_file:
            specific_req_path = os.path.join(REQUIREMENTS_DIR, specific_file)
            requirements.extend(self._read_requirements_file(specific_req_path))

        return list(dict.fromkeys(requirements))

    def log_model(
        self,
        model,
        artifact_path: str = "model",
        framework: str = "sklearn",
        signature=None,
        input_example=None,
        model_name: Optional[str] = None,
    ):
        """
        Đóng gói mô hình theo MLflow flavor cùng signature, input example và dependencies.
        """
        pip_reqs = self._get_pip_requirements(framework)

        try:
            flavor = {
                "sklearn": sklearn,
                "xgboost": xgboost,
                "pytorch": pytorch,
            }.get(framework)
            if flavor is None:
                raise ValueError(f"Unsupported MLflow model framework: {framework}")

            log_model_parameters = inspect.signature(flavor.log_model).parameters
            model_options: Dict[str, Any] = (
                {"name": model_name or artifact_path.replace("/", "_")}
                if "name" in log_model_parameters
                else {"artifact_path": artifact_path}
            )
            if framework == "sklearn" and "skops_trusted_types" in log_model_parameters:
                model_options["skops_trusted_types"] = ["numpy.dtype"]
            return flavor.log_model(
                model,
                **model_options,
                signature=signature,
                input_example=input_example,
                pip_requirements=pip_reqs,
            )
        except Exception as e:
            logger.error(f"Lỗi khi đóng gói mô hình: {e}")
            raise
        logger.info(
            f"Đã đóng gói mô hình vào '{artifact_path}' kèm môi trường chuẩn "
            f"từ requirements ({len(pip_reqs)} packages)."
        )

    def log_validation_errors(self, schema_name: str, errors: List[Dict[str, Any]]):
        """
        Lưu danh sách lỗi Validation thành file JSON cục bộ tại thư mục Artifacts,
        sau đó đẩy lên MLflow để tracking.
        """
        if not errors:
            return

        try:
            # 1. Xác định đường dẫn thư mục Artifacts nội bộ
            # PROJECT_ROOT đã được khai báo ở đầu file mlflow_tracker.py
            validation_dir = os.path.join(
                PROJECT_ROOT, "Artifacts", "reports", "ml_calories", "validation"
            )
            os.makedirs(validation_dir, exist_ok=True)

            # Đặt tên file (ví dụ: CaloriesSchema_val_report.json)
            local_val_report_path = os.path.join(
                validation_dir, f"{schema_name}_val_report.json"
            )

            # 2. Lưu file JSON cục bộ
            with open(local_val_report_path, "w", encoding="utf-8") as f:
                json.dump(errors, f, ensure_ascii=False, indent=4)

            logger.info(f"Đã lưu báo cáo lỗi cục bộ tại: {local_val_report_path}")

            # 3. Đồng bộ file nội bộ này lên MLflow Artifacts
            mlflow.log_artifact(
                local_val_report_path, artifact_path=f"validation_reports/{schema_name}"
            )
            logger.info(f"Đã đồng bộ báo cáo lỗi '{schema_name}' lên MLflow Artifacts.")

        except Exception as e:
            logger.error(f"Lỗi khi lưu/upload validation report: {e}")

    def end_run(self, status: str = "FINISHED"):
        """
        Kết thúc MLflow run hiện tại, đồng bộ log hệ thống và thiết lập trạng thái.
        """
        try:
            # Tự động upload log file lên artifact trước khi đóng run
            self.upload_system_logs()

            # Kết thúc run với trạng thái tương ứng (FINISHED, FAILED, KILLED)
            mlflow.end_run(status=status)
            logger.info(f"Đã kết thúc MLflow run với trạng thái: {status}")
        except Exception as e:
            logger.error(f"Lỗi khi kết thúc MLflow run: {e}")

    def log_artifact_file(self, local_path: str, artifact_path: Optional[str] = None):
        """Upload một file bất kỳ (như báo cáo, biểu đồ, file lỗi) lên MLflow."""
        try:
            if os.path.exists(local_path):
                mlflow.log_artifact(local_path, artifact_path=artifact_path)
                logger.info(f"Đã upload {local_path} lên MLflow Artifacts.")
            else:
                logger.warning(f"Không tìm thấy file để upload: {local_path}")
        except Exception as e:
            logger.error(f"Lỗi khi upload artifact {local_path}: {e}")


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
