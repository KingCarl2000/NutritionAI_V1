-- ============================================================================
-- PHẦN 1: CÁC LỆNH HỆ THỐNG CỦA PSQL (INTERACTIVE TERMINAL)
-- LƯU Ý: Các lệnh này chỉ chạy được trong giao diện dòng lệnh psql.
-- ============================================================================

-- Hiển thị danh sách cột, kiểu dữ liệu, Nullable, Default, index và khóa ngoại:
-- \d fitness_tracker_dataset

-- Cung cấp thông số chi tiết nâng cao (comments, kích thước TOAST, RLS...):
-- \d+ fitness_tracker_dataset


-- ============================================================================
-- PHẦN 2: TRUY VẤN TỪ HỆ THỐNG ANSI-COMPLIANT INFORMATION_SCHEMA.COLUMNS
-- Sử dụng khi kết nối từ ứng dụng bên ngoài (Python, Jupyter, DBeaver...)
-- ============================================================================
SELECT column_name, data_type, is_nullable, column_default, character_maximum_length, numeric_precision, numeric_scale 
FROM information_schema.columns 
WHERE table_schema = 'raw'            
    AND table_name = 'fitness_tracker_dataset'  
ORDER BY ordinal_position;


-- ============================================================================
-- PHẦN 3: TRUY XUẤT CHUYÊN SÂU QUA SYSTEM CATALOGS (pg_attribute, pg_class)
-- Dùng để tự động phân tích siêu dữ liệu (metadata profiling) phức tạp
-- ============================================================================
SELECT 
    a.attnum AS column_id,
    a.attname AS column_name,
    format_type(a.atttypid, a.atttypmod) AS data_type,
    a.attnotnull AS is_not_null,
    col_description(a.attrelid, a.attnum) AS column_comment
FROM 
    pg_attribute a
WHERE 
    a.attrelid = 'raw.fitness_tracker_dataset'::regclass
    AND a.attnum > 0              
    AND NOT a.attisdropped        
ORDER BY 
    a.attnum;


-- ============================================================================
-- PHẦN 4: KIỂM TRA CÁC RÀNG BUỘC DỮ LIỆU (DATA CONSTRAINTS)
-- Phát hiện các quy tắc kiểm tra tính toàn vẹn và miền giá trị của dữ liệu
-- ============================================================================
SELECT 
    conname AS constraint_name,
    contype AS constraint_type,   
    pg_get_constraintdef(oid) AS constraint_definition
FROM 
    pg_constraint
WHERE 
    conrelid = 'raw.fitness_tracker_dataset'::regclass;


-- ============================================================================
-- PHẦN 5: KIỂM TRA CẤU TRÚC PHÂN VÙNG (PARTITION TREE VERIFICATION)
-- Hiểu cách phân bổ dữ liệu và thống kê dung lượng lưu trữ thực tế
-- ============================================================================

-- (Tùy chọn) Hiển thị cấu trúc khóa phân vùng đã dùng để thiết lập bảng cha:
-- SELECT pg_get_partkeydef('raw.fitness_tracker_dataset'::regclass);

-- Thống kê chi tiết dung lượng lưu trữ thực tế của từng phân vùng con:
SELECT 
    p.relid::regclass AS partition_name,
    p.level,
    p.isleaf,
    pg_size_pretty(pg_total_relation_size(p.relid)) AS total_size
FROM 
    pg_partition_tree('raw.fitness_tracker_dataset'::regclass) p;


-- ============================================================================
-- PHẦN 6: ƯỚC LƯỢNG SỐ DÒNG VÀ KÍCH THƯỚC BẢNG (SIZE & ROW ESTIMATION)
-- Tối ưu hiệu năng khi phân tích bảng lớn thay vì dùng COUNT(*)
-- ============================================================================

-- 6.1 Ước lượng số dòng thông qua catalog pg_class (nhanh hơn COUNT(*) rất nhiều)
SELECT relname AS relation_name, relpages AS disk_pages, reltuples AS estimated_rows 
FROM pg_class 
WHERE oid = 'raw.fitness_tracker_dataset'::regclass;

-- 6.2 Tổng quan kích thước vật lý của bảng chuẩn (bao gồm TOAST, Index)
SELECT 
    pg_size_pretty(pg_table_size('raw.fitness_tracker_dataset'::regclass)) AS table_size,
    pg_size_pretty(pg_indexes_size('raw.fitness_tracker_dataset'::regclass)) AS index_size,
    pg_size_pretty(pg_total_relation_size('raw.fitness_tracker_dataset'::regclass)) AS total_size;

-- 6.3 Ước lượng tổng kích thước đối với bảng phân vùng (Partitioned Tables)
SELECT 
    pg_size_pretty(sum(pg_relation_size(relid))) AS total_partition_size 
FROM 
    pg_partition_tree('raw.fitness_tracker_dataset'::regclass);

/*
-- phần 7: Xóa các cột không cần thiết để giảm dung lượng lưu trữ và tăng tốc độ truy vấn
ALTER TABLE raw.fitness_tracker_dataset
DROP COLUMN IF EXISTS "m_creationTime_am_pm",
DROP COLUMN IF EXISTS "m_startTime_am_pm",
DROP COLUMN IF EXISTS "m_endTime_am_pm",
DROP COLUMN IF EXISTS "m_creationDate",
DROP COLUMN IF EXISTS "m_creationTime",
DROP COLUMN IF EXISTS "m_creationTimeZone";
*/

-- ============================================================================
-- PHẦN 8: PHÁC THẢO PHÂN PHỐI DỮ LIỆU (STATISTICAL PROFILING)
-- Khám phá đặc trưng thống kê của dữ liệu mà không cần quét lại toàn bộ bảng
-- ============================================================================

-- 8.1 Truy xuất thống kê chi tiết từ catalog pg_stats
SELECT 
    attname AS column_name,
    null_frac AS missing_ratio, -- Tỷ lệ dữ liệu NULL 
    n_distinct,                 -- Độ phân biệt 
    avg_width AS avg_byte_width,-- Dung lượng RAM trung bình (bytes) 
    most_common_vals,           -- Mảng các giá trị phổ biến nhất 
    most_common_freqs,          -- Tần suất tương ứng của các giá trị phổ biến
    histogram_bounds[1] AS estimated_min, -- Trích xuất Min từ mảng phân phối
    histogram_bounds[array_length(histogram_bounds, 1)] AS estimated_max, -- Trích xuất Max từ mảng phân phối
    correlation                 -- Hệ số tương quan vật lý/logic trên đĩa
FROM 
    pg_stats 
WHERE 
    tablename = 'fitness_tracker_dataset' AND schemaname = 'raw'
ORDER BY 
    attname;

-- 8.2 Tinh chỉnh độ chi tiết của thống kê (Tùy chọn)
-- ALTER TABLE raw.fitness_tracker_dataset ALTER COLUMN calories SET STATISTICS 500;
-- ANALYZE raw.fitness_tracker_dataset;


-- ============================================================================
-- PHẦN 9: LẤY MẪU DỮ LIỆU ĐẠI DIỆN (DATA SAMPLING)
-- Trích xuất nhanh dữ liệu để tải vào Pandas/Jupyter mà không làm tràn RAM
-- ============================================================================

-- 9.1 Lấy mẫu bằng hàm mở rộng hệ thống (Yêu cầu bật extension tsm_system_rows, tsm_system_time)
SELECT * FROM raw.fitness_tracker_dataset TABLESAMPLE SYSTEM_ROWS(100);

-- 9.2 Lấy mẫu ngẫu nhiên lặp lại (Reproducible Sampling cho Machine Learning)
SELECT setseed(0.5); 
SELECT * FROM raw.fitness_tracker_dataset WHERE random() < 0.01;

-- 9.4 Lấy mẫu phân trang/giới hạn an toàn bằng LIMIT và OFFSET
SELECT * FROM raw.fitness_tracker_dataset 
ORDER BY 
    "user_id", 
    "date"
LIMIT 100 OFFSET 1000;


-- ============================================================================
-- PHẦN 10: XUẤT KẾT QUẢ RA FILE (EXPORTING DATA)
-- Chạy các lệnh này trong psql terminal để lưu kết quả EDA / Tập lấy mẫu
-- ============================================================================

-- Tắt phân trang (paging) để dữ liệu xuất ra không bị ngắt quãng bởi '--- More ---'
\pset pager off

-- Bước 1: Chuyển hướng đầu ra vào file báo cáo
\o 'D:/NutritionAI_V1/Artifacts/schema_profiling_report.txt'

-- Bước 2: Tạo tiêu đề báo cáo
SELECT '===================================================' AS " ";
SELECT '  BÁO CÁO KHÁM PHÁ SIÊU DỮ LIỆU (METADATA PROFILING) ' AS " ";
SELECT '  Bảng dữ liệu nguồn: raw.fitness_tracker_dataset' AS " ";
SELECT '  Thời gian khởi tạo: ' || current_timestamp AS " ";
SELECT '===================================================' AS " ";

-- Bước 3: Xuất cấu trúc Schema chi tiết [Mục 35.17]
SELECT '1. CẤU TRÚC SCHEMA BẢNG' AS "---";
SELECT 
    column_name AS "Column Name", 
    data_type AS "Data Type", 
    is_nullable AS "Allows Null"
FROM information_schema.columns 
WHERE table_schema = 'raw' AND table_name = 'fitness_tracker_dataset'
ORDER BY ordinal_position;

-- Bước 4: Xuất ước lượng kích thước vật lý và số lượng dòng [Mục 52.11 / 9.28.7]
SELECT '2. ƯỚC LƯỢNG KÍCH THƯỚC VẬT LÝ' AS "---";
SELECT 
    reltuples::bigint AS "Số dòng ước tính",
    pg_size_pretty(pg_relation_size('raw.fitness_tracker_dataset'::regclass)) AS "Dung lượng bảng thô",
    pg_size_pretty(pg_total_relation_size('raw.fitness_tracker_dataset'::regclass)) AS "Tổng dung lượng (gồm Chỉ mục)"
FROM pg_class 
WHERE oid = 'raw.fitness_tracker_dataset'::regclass;

-- Bước 5: Xuất phác thảo phân phối của các cột số [Mục 53.29]
SELECT '3. THỐNG KÊ PHÂN PHỐI DỮ LIỆU TỪ PG_STATS' AS "---";
SELECT 
    attname AS "Column Name",
    null_frac AS "Null Ratio",
    n_distinct AS "Distinct Count",
    (histogram_bounds::text::text[])[1] AS "Estimated Min",
    (histogram_bounds::text::text[])[array_length(histogram_bounds::text::text[], 1)] AS "Estimated Max",
    avg_width AS "Average Byte Width (RAM)"
FROM pg_stats
WHERE tablename = 'fitness_tracker_dataset' AND schemaname = 'raw'
ORDER BY attname;

-- Bước 6: Đóng file báo cáo, quay lại console hiển thị và bật lại phân trang
\o
\pset pager on
