"""
Module chứa các Custom Check Generators & Extensions tổng quát cho Pandera.
Tất cả các hàm trong module này hoàn toàn độc lập với dataset, chỉ đóng vai trò
là công cụ (Utilities) để các Pipeline Schema tùy chỉnh truyền tham số và sử dụng.
"""
import re
from typing import List, Optional, Any
import pandas as pd
import pandera.pandas as pa
import pandera.extensions as pa_ext


# ==============================================================================
# 1. EXTENSIONS API (Đăng ký phương thức kiểm tra Vectorized dùng chung)
# ==============================================================================

@pa_ext.register_check_method(
    statistics=["min_val", "max_val"],
    check_type="vectorized"
)
def is_in_numeric_range(
    pandas_obj: pd.Series, 
    *, 
    min_val: float = 0.0, 
    max_val: float = 1e9
) -> pd.Series:
    """Kiểm tra một cột số nằm trong khoảng [min_val, max_val]."""
    return (pandas_obj >= min_val) & (pandas_obj <= max_val)


@pa_ext.register_check_method(
    statistics=["min_len", "max_len"],
    check_type="vectorized"
)
def is_valid_string_length(
    pandas_obj: pd.Series, 
    *, 
    min_len: int = 1, 
    max_len: int = 255
) -> pd.Series:
    """Kiểm tra độ dài chuỗi ký tự nằm trong khoảng [min_len, max_len]."""
    str_lens = pandas_obj.astype(str).str.len()
    return (str_lens >= min_len) & (str_lens <= max_len)


# ==============================================================================
# 2. FACTORY FUNCTIONS (Tạo các quy tắc Wide / Groupby / Warning tổng quát)
# ==============================================================================

class GenericCheckFactory:
    """Tập hợp các hàm khởi tạo quy tắc kiểm định đa cột và nâng cao dạng tham số hóa."""

    @staticmethod
    def check_sum_le_total(
        component_cols: List[str],
        total_col: str,
        error_msg: Optional[str] = None
    ) -> pa.Check:
        """
        [Wide Check] Kiểm tra tổng giá trị của danh sách các cột thành phần 
        phải nhỏ hơn hoặc bằng giá trị của cột tổng.
        """
        default_err = f"Tổng các cột {component_cols} vượt quá giá trị cột tổng '{total_col}'."
        return pa.Check(
            lambda df: df[component_cols].sum(axis=1) <= df[total_col],
            error=error_msg or default_err,
            ignore_na=True
        )

    @staticmethod
    def check_range_validity(
        min_col: str,
        target_col: str,
        max_col: str,
        error_msg: Optional[str] = None
    ) -> pa.Check:
        """
        [Wide Check] Kiểm tra logic thứ tự giữa 3 cột: min_col <= target_col <= max_col.
        """
        default_err = f"Ràng buộc không thỏa mãn: '{min_col}' <= '{target_col}' <= '{max_col}'."
        return pa.Check(
            lambda df: (df[min_col] <= df[target_col]) & (df[target_col] <= df[max_col]),
            error=error_msg or default_err,
            ignore_na=True
        )

    @staticmethod
    def check_group_mean_comparison(
        groupby_col: str,
        target_col: str,
        lower_group: Any,
        higher_group: Any,
        error_msg: Optional[str] = None
    ) -> pa.Check:
        """
        [Groupby Check] Kiểm tra trung bình cột target_col của lower_group 
        phải nhỏ hơn trung bình của higher_group.
        """
        default_err = f"Trung bình '{target_col}' của nhóm '{lower_group}' không nhỏ hơn nhóm '{higher_group}'."
        return pa.Check(
            lambda g: g.get(lower_group, pd.Series([0]))[target_col].mean() < 
                       g.get(higher_group, pd.Series([float("inf")]))[target_col].mean(),
            groupby=groupby_col,
            error=error_msg or default_err,
            ignore_na=True
        )

    @staticmethod
    def check_regex_pattern(
        pattern: str,
        error_msg: Optional[str] = None
    ) -> pa.Check:
        """
        [Element-wise Check] Kiểm tra định dạng chuỗi của từng phần tử theo biểu thức Regex.
        """
        compiled_regex = re.compile(pattern)
        default_err = f"Giá trị không khớp với định dạng Regex chuẩn: '{pattern}'."
        return pa.Check(
            lambda val: bool(compiled_regex.match(str(val))),
            element_wise=True,
            error=error_msg or default_err,
            ignore_na=True
        )

    @staticmethod
    def create_outlier_warning(
        upper_threshold: float,
        warning_msg: Optional[str] = None
    ) -> pa.Check:
        """
        [Soft Check] Phát cảnh báo SchemaWarning khi giá trị vượt ngưỡng cực đại (Outlier) 
        mà không dừng chương trình.
        """
        default_msg = f"CẢNH BÁO: Phát hiện giá trị nghi ngờ là Outlier (> {upper_threshold})."
        return pa.Check(
            lambda series: series <= upper_threshold,
            raise_warning=True,
            error=warning_msg or default_msg
        )