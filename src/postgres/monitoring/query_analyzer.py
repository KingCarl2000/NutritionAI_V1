import logging
import json
from typing import LiteralString, cast
from psycopg.sql import SQL
from src.postgres.core.connection import get_connection

logger = logging.getLogger(__name__)

def analyze_query_performance(query: str):
    """
    Tự động chèn EXPLAIN (ANALYZE, FORMAT JSON) vào truy vấn Feature Store
    để trích xuất các thông tin về thời gian thực thi và phát hiện Seq Scan.
    """
    explain_query = f"EXPLAIN (ANALYZE, FORMAT JSON) {query}"
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(SQL(cast(LiteralString, explain_query)))
            
            row = cur.fetchone()
            if not row or not row[0]:
                logger.warning("Không thể trích xuất execution plan từ câu lệnh EXPLAIN.")
                return None
                
            plan_json = row[0]
            
            plan = plan_json[0]['Plan']
            execution_time = plan_json[0].get('Execution Time', 0)
            
            logger.info(f"Thời gian thực thi (Execution Time): {execution_time} ms")
            
            def check_seq_scan(node):
                if node.get('Node Type') == 'Seq Scan':
                    table = node.get('Relation Name')
                    logger.warning(f"CẢNH BÁO: Phát hiện 'Seq Scan' (Quét toàn bảng) trên bảng '{table}'. Cần tạo Index ngay!")
                
                if 'Plans' in node:
                    for sub_plan in node['Plans']:
                        check_seq_scan(sub_plan)
                        
            check_seq_scan(plan)
            return plan_json