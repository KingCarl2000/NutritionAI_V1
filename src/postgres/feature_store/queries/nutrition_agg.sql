-- ==============================================================================
-- FILE: nutrition_agg.sql
-- MỤC ĐÍCH: Trích xuất đặc trưng (Feature Extraction) từ dữ liệu fitness_logs
-- áp dụng Hàm gộp (Aggregate Functions) và Hàm cửa sổ (Window Functions).
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- PHẦN 1: TẠO ĐẶC TRƯNG TỔNG HỢP (AGGREGATE FEATURES)
-- Sử dụng mệnh đề FILTER để tạo nhiều đặc trưng đếm/tổng trên một lần quét dữ liệu.
-- ------------------------------------------------------------------------------
CREATE OR REPLACE VIEW user_workout_summary AS
SELECT 
    user_id,
    COUNT(*) AS total_workouts,
    -- Đếm số ngày tập nặng (calo > 500)
    COUNT(*) FILTER (WHERE calories_burned > 500) AS heavy_workout_days,
    -- Đếm số ngày tập nhẹ (calo <= 500)
    COUNT(*) FILTER (WHERE calories_burned <= 500) AS light_workout_days,
    -- Tính cân nặng trung bình
    ROUND(AVG(weight)::numeric, 2) AS avg_weight
FROM fitness_logs
GROUP BY user_id
HAVING COUNT(*) > 10; -- Lọc bỏ những người dùng tập ít hơn 10 buổi

-- ------------------------------------------------------------------------------
-- PHẦN 2 & 3: TẠO ĐẶC TRƯNG CHUỖI THỜI GIAN VÀ LỌC BẰNG SUBQUERY
-- Áp dụng Window Functions để tính trung bình trượt, độ trễ và xếp hạng.
-- Bọc bên trong một Subquery (hoặc CTE - Common Table Expression) để lọc top 3.
-- ------------------------------------------------------------------------------
CREATE OR REPLACE VIEW top_user_workout_features AS
WITH WindowedFeatures AS (
    SELECT 
        user_id,
        log_date,
        weight,
        calories_burned,
        
        -- Đặc trưng 1: Trung bình trượt (Moving Average) calo trong 7 buổi tập gần nhất
        ROUND(AVG(calories_burned) OVER (
            PARTITION BY user_id 
            ORDER BY log_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        )::numeric, 2) AS moving_avg_7d_calories,
        
        -- Đặc trưng 2: Lượng calo của buổi tập liền trước (Lag) để theo dõi xu hướng
        LAG(calories_burned, 1) OVER (
            PARTITION BY user_id 
            ORDER BY log_date
        ) AS prev_session_calories,
        
        -- Đặc trưng 3: Xếp hạng (Rank) ngày tiêu thụ calo nhiều nhất của người dùng
        ROW_NUMBER() OVER (
            PARTITION BY user_id 
            ORDER BY calories_burned DESC
        ) AS calorie_burn_rank
        
    FROM fitness_logs
)
-- Truy vấn chính: Lọc dựa trên kết quả của Window Function
SELECT 
    user_id,
    log_date,
    weight,
    calories_burned,
    moving_avg_7d_calories,
    prev_session_calories,
    calorie_burn_rank,
    -- Tính xu hướng: lượng calo thay đổi so với buổi trước
    (calories_burned - prev_session_calories) AS calorie_diff_from_prev
FROM WindowedFeatures
WHERE calorie_burn_rank <= 3; -- Chỉ giữ lại Top 3 ngày đốt nhiều calo nhất của mỗi người dùng