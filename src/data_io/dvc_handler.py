import os
import sys
import json
import base64
import subprocess
import joblib
import pandas as pd
from typing import Any, Optional, Dict
from pydantic import BaseModel, Field, ConfigDict
import dvc.api

# --- Tái sử dụng Logging và Exception từ hệ thống ---
from src.nutrition_core.logging.logger import logger, dvc_error_handler
from src.nutrition_core.exception.exception import DataPipelineException

# ==========================================
# 1. Pydantic Schemas cho Interface Adaptation (Đã thiết kế lại)
# ==========================================
class PerClassMetric(BaseModel):
    model_config = ConfigDict(extra="allow")
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    support: int = Field(..., ge=0)

class GlobalSummaryMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")
    accuracy: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    loss: float = Field(..., ge=0.0)
    roc_auc: Optional[float] = Field(None, ge=0.0, le=1.0)

class EpochMetrics(BaseModel):
    best_epoch: int = Field(..., ge=1)
    total_epochs: int = Field(..., ge=1)
    train_loss_final: float = Field(..., ge=0.0)
    val_loss_final: float = Field(..., ge=0.0)

class ExecutionMetadata(BaseModel):
    dataset_version: Optional[str] = None
    execution_time_seconds: float = Field(..., ge=0.0)
    num_samples_eval: int = Field(..., ge=1)
    framework: Optional[str] = None

class DVCMetricsNormalizedSchema(BaseModel):
    """Lược đồ dữ liệu chuẩn hóa cấp cao nhất cho tệp metrics.json từ DVC"""
    summary: GlobalSummaryMetrics
    class_metrics: Optional[Dict[str, PerClassMetric]] = None
    epoch_metrics: Optional[EpochMetrics] = None
    metadata: Optional[ExecutionMetadata] = None


# ==========================================
# 2. Custom Exception cho DVC
# ==========================================
class DVCIntegrationError(DataPipelineException):
    def __init__(self, message: str):
        super().__init__(error_message=message, error_details=sys)


# ==========================================
# 3. Adapter Class: DVCHandler
# ==========================================
class DVCHandler:
    """
    Adapter che giấu toàn bộ sự phức tạp của DVC, xử lý đồng bộ, 
    và chuẩn hóa dữ liệu trả về cho lõi nghiệp vụ.
    """

    def __init__(self, repo: Optional[str] = None, rev: Optional[str] = None):
        self.repo = repo
        self.rev = rev
        logger.info(f"Khởi tạo DVCHandler (repo={repo}, rev={rev}). Sẵn sàng kết nối Remote Storage.")

    def sync_data(self) -> None:
        logger.info("Đang đồng bộ dữ liệu qua DVC pull...")
        try:
            result = subprocess.run(["dvc", "pull"], capture_output=True, text=True, check=True)
            logger.info(f"Đồng bộ thành công: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            logger.error(f"[DVC ERROR] Không thể sync data: {e.stderr}")
            raise DVCIntegrationError(f"Lỗi khi đồng bộ dữ liệu (dvc pull): {e.stderr}")

    def is_data_updated(self) -> bool:
        logger.info("Kiểm tra trạng thái cập nhật của dữ liệu...")
        try:
            result = subprocess.run(["dvc", "status", "-c"], capture_output=True, text=True, check=True)
            return "up to date" in result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"[DVC ERROR] Không thể kiểm tra trạng thái: {e.stderr}")
            raise DVCIntegrationError("Lỗi khi kiểm tra trạng thái DVC.")

    def get_dataset(self, data_path: str, format_type: str = 'csv') -> pd.DataFrame:
        logger.info(f"Đang tải dataset từ {data_path} (định dạng {format_type})...")
        try:
            with dvc.api.open(data_path, repo=self.repo, rev=self.rev, mode='r') as f:
                if format_type == 'csv':
                    df = pd.read_csv(f)
                elif format_type == 'parquet':
                    with dvc.api.open(data_path, repo=self.repo, rev=self.rev, mode='rb') as f_bin:
                        df = pd.read_parquet(f_bin)
                else:
                    raise ValueError(f"Định dạng {format_type} chưa được hỗ trợ.")
            
            logger.info(f"Tải thành công dataset {data_path} với shape {df.shape}.")
            return df
        except Exception as e:
            logger.error(f"Lỗi khi tải dataset {data_path}: {str(e)}")
            raise DVCIntegrationError(f"Không thể đọc dataset từ DVC: {str(e)}")

    def get_latest_model(self, model_path: str, framework: str = 'sklearn') -> Any:
        logger.info(f"Đang tải mô hình {framework} từ {model_path}...")
        try:
            with dvc.api.open(model_path, repo=self.repo, rev=self.rev, mode='rb') as f:
                if framework == 'sklearn':
                    model = joblib.load(f)
                elif framework == 'pytorch':
                    import torch
                    model = torch.load(f)
                else:
                    raise ValueError(f"Framework {framework} chưa được hỗ trợ.")
            logger.info("Tải mô hình thành công.")
            return model
        except Exception as e:
            logger.error(f"Lỗi khi stream model từ DVC: {str(e)}")
            raise DVCIntegrationError(f"Không thể load model từ DVC: {str(e)}")

    def get_metrics(self, metrics_path: str = "metrics.json") -> DVCMetricsNormalizedSchema:
        """
        Interface Adaptation: Đọc JSON từ DVC và map trực tiếp vào thiết kế 
        Schema chuẩn hóa (DVCMetricsNormalizedSchema) chứa validations.
        """
        logger.info(f"Đọc metrics từ {metrics_path}...")
        try:
            with dvc.api.open(metrics_path, repo=self.repo, rev=self.rev, mode='r') as f:
                data = json.load(f)
                
                # Khởi tạo Schema phân cấp. Nếu data thiếu field hoặc sai kiểu,
                # Pydantic ValidationError sẽ được văng ra tự động.
                metrics_dto = DVCMetricsNormalizedSchema(**data)
                
                logger.info(f"Phân tích file metrics thành công: {metrics_dto.summary.accuracy}")
                return metrics_dto
                
        except Exception as e:
            logger.error(f"Lỗi khi đọc hoặc xác thực file metrics: {str(e)}")
            raise DVCIntegrationError(f"Không thể phân tích file metrics: {str(e)}")

    def get_image_base64(self, image_path: str) -> str:
        logger.info(f"Tải ảnh từ {image_path} và chuyển đổi sang Base64...")
        try:
            with dvc.api.open(image_path, repo=self.repo, rev=self.rev, mode='rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Lỗi khi xử lý ảnh từ DVC: {str(e)}")
            raise DVCIntegrationError(f"Không thể lấy dữ liệu ảnh: {str(e)}")