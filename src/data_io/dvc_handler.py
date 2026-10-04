import os
import json
import base64
import subprocess
import joblib
import pandas as pd
from typing import Any, Optional, Dict
from pydantic import BaseModel, Field, ConfigDict
import dvc.api
import torch

# --- Import Logger mới cập nhật chứa Decorator xử lý lỗi ---
from src.nutrition_core.logging.logger import logger, dvc_error_handler

# ==========================================
# 1. Pydantic Schemas cho Interface Adaptation
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
    summary: GlobalSummaryMetrics
    class_metrics: Optional[Dict[str, PerClassMetric]] = None
    epoch_metrics: Optional[EpochMetrics] = None
    metadata: Optional[ExecutionMetadata] = None


# ==========================================
# 2. Adapter Class: DVCHandler
# ==========================================
class DVCHandler:
    """
    Adapter che giấu sự phức tạp của DVC.
    Đã tích hợp xử lý lỗi tập trung qua Decorator @dvc_error_handler.
    """

    def __init__(self, repo: Optional[str] = None, rev: Optional[str] = None):
        self.repo = repo
        self.rev = rev
        logger.info(f"Khởi tạo DVCHandler (repo={self.repo}, rev={self.rev}).")

    # --- Cơ chế dự phòng (Fallback) ---
    def _fallback_local_dataset(self, data_path: str, format_type: str = 'csv', **kwargs) -> pd.DataFrame:
        """Đọc file từ thư mục local/cache nếu không thể kết nối tới DVC Remote"""
        file_name = os.path.basename(data_path)
        local_path = os.path.join("data", "raw", file_name)
        logger.info(f"[FALLBACK] Thử đọc dữ liệu từ local cache: {local_path}")
        
        if format_type == 'csv':
            return pd.read_csv(local_path)
        return pd.read_parquet(local_path)

    # --- Các hàm nghiệp vụ chính ---

    @dvc_error_handler(max_retries=3, retry_delay=2)
    def sync_data(self) -> None:
        """Kéo dữ liệu từ remote storage (che giấu dvc pull)"""
        logger.info("Đang đồng bộ dữ liệu qua DVC pull...")
        result = subprocess.run(["dvc", "pull"], capture_output=True, text=True, check=True)
        logger.info(f"Đồng bộ thành công: {result.stdout.strip()}")

    @dvc_error_handler(max_retries=1)
    def is_data_updated(self) -> bool:
        """Kiểm tra trạng thái cập nhật của dữ liệu (che giấu dvc status)"""
        logger.info("Kiểm tra trạng thái cập nhật của dữ liệu...")
        result = subprocess.run(["dvc", "status", "-c"], capture_output=True, text=True, check=True)
        return "up to date" in result.stdout

    @dvc_error_handler(max_retries=3, retry_delay=2, fallback_func=_fallback_local_dataset)
    def get_dataset(self, data_path: str, format_type: str = 'csv') -> pd.DataFrame:
        """Stream tệp dữ liệu từ DVC và trả về Pandas DataFrame."""
        logger.info(f"Đang tải dataset từ {data_path} (định dạng {format_type})...")
        
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

    @dvc_error_handler(max_retries=2, retry_delay=3)
    def get_latest_model(self, model_path: str, framework: str = 'sklearn') -> Any:
        """Tải mô hình từ DVC Remote và chuyển đổi thành Object."""
        logger.info(f"Đang tải mô hình {framework} từ {model_path}...")
        
        with dvc.api.open(model_path, repo=self.repo, rev=self.rev, mode='rb') as f:
            if framework == 'sklearn':
                model = joblib.load(f)
            elif framework == 'pytorch':
                model = torch.load(f)
            else:
                raise ValueError(f"Framework {framework} chưa được hỗ trợ.")
            
        logger.info("Tải mô hình thành công.")
        return model

    @dvc_error_handler(max_retries=1)
    def get_metrics(self, metrics_path: str = "metrics.json") -> DVCMetricsNormalizedSchema:
        """Đọc file metrics.json và validate qua Pydantic Schema."""
        logger.info(f"Đọc metrics từ {metrics_path}...")
        
        with dvc.api.open(metrics_path, repo=self.repo, rev=self.rev, mode='r') as f:
            data = json.load(f)
            metrics_dto = DVCMetricsNormalizedSchema(**data)
            logger.info(f"Phân tích file metrics thành công: {metrics_dto.summary.accuracy}")
            return metrics_dto

    @dvc_error_handler(max_retries=2, retry_delay=1)
    def get_image_base64(self, image_path: str) -> str:
        """Đọc ảnh từ DVC và chuyển đổi sang chuỗi Base64."""
        logger.info(f"Tải ảnh từ {image_path} và chuyển sang Base64...")
        
        with dvc.api.open(image_path, repo=self.repo, rev=self.rev, mode='rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')