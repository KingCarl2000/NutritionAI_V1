import psycopg
from psycopg import IsolationLevel
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureInferenceDB:
    def __init__(self, standby_conn_info: str):
        """
        Khởi tạo kết nối ĐỌC (Read-Only) đến máy chủ bản sao Hot Standby.
        """
        self.standby_conn_info = standby_conn_info

    def fetch_realtime_features(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Truy xuất các đặc trưng thời gian thực cho một user cụ thể phục vụ mô hình ML.
        """
        # Giới hạn mức cô lập giao dịch: Hot Standby chỉ hỗ trợ tối đa REPEATABLE READ.
        # (Serializable không được hỗ trợ trên bản sao).
        try:
            with psycopg.connect(
                self.standby_conn_info,
                isolation_level=IsolationLevel.REPEATABLE_READ, 
                autocommit=False
            ) as conn:
                
                # Bật row_factory để trả về dictionary dễ dàng dùng cho Pandas hoặc Model
                conn.row_factory = psycopg.rows.dict_row
                
                with conn.cursor() as cursor:
                    # LƯU Ý VỀ QUERY CONFLICTS (Xung đột truy vấn):
                    # Để tránh truy vấn ML bị hủy do máy Primary dọn dẹp (VACUUM) các dòng cũ,
                    # quản trị viên cần thiết lập 'hot_standby_feedback = on' 
                    # hoặc cấu hình 'max_standby_streaming_delay' trong postgresql.conf của Standby.
                    
                    logger.info(f"Đang fetch feature vector mới nhất (Eventual Consistency) cho User {user_id}...")
                    
                    # Truy vấn View đặc trưng (từ bài toán Hàm cửa sổ/Hàm gộp trước đó)
                    query = """
                        SELECT 
                            user_id, 
                            weight, 
                            calories_burned, 
                            moving_avg_7d_calories, 
                            prev_session_calories
                        FROM top_user_workout_features
                        WHERE user_id = %s
                        ORDER BY log_date DESC
                        LIMIT 1;
                    """
                    cursor.execute(query, (user_id,))
                    feature_vector = cursor.fetchone()
                    
                    if feature_vector:
                        logger.info("Đã trích xuất thành công features từ Hot Standby.")
                        return feature_vector
                    else:
                        logger.warning("Không tìm thấy dữ liệu hoặc chưa được luân chuyển tới Standby.")
                        return None
                        
        except psycopg.Error as e:
            logger.error(f"Lỗi truy xuất trên Hot Standby (có thể do Query Conflict): {e}")
            raise e

# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    # Chuỗi kết nối chỉ hướng tới REPLICA/HOT STANDBY NODE
    db_url_standby = "postgresql://ml_readonly:password@standby_host:5432/nutrition_db"
    
    inference_db = FeatureInferenceDB(db_url_standby)
    features = inference_db.fetch_realtime_features(user_id=101)
    
    if features:
        print("Feature Vector chuẩn bị đưa vào Model:", features)
        # Tại đây: Truyền `features` vào model ML (vd: XGBoost, PyTorch) để predict()