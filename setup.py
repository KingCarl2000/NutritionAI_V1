"""
The setup.py file is an essential part of packaging and 
distributing Python projects.
"""

from setuptools import find_packages, setup
from typing import List
import os

def get_requirements(file_path: str) -> List[str]:
    """
    This function returns a list of requirements from a specified text file.
    """
    requirement_lst: List[str] = []
    
    if not os.path.exists(file_path):
        print(f"Requirement file not found at: {file_path}")
        return requirement_lst

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            requirement = line.strip()
            # Bỏ qua dòng trống, comment và '-e .'
            if requirement and not requirement.startswith("#") and requirement != "-e .":
                requirement_lst.append(requirement)

    return requirement_lst

# Đọc danh sách từ từng file
base_reqs = get_requirements("requirements/base.txt")
ml_reqs = get_requirements("requirements/ml.txt")
llm_reqs = get_requirements("requirements/llm.txt")

setup(
    name="NutritionAI_V1",
    version="0.0.1",
    author="Carl",
    author_email="ngvcuong282@gmail.com",
    packages=find_packages(),
    # install_requires cài đặt mặc định (chỉ lấy base)
    install_requires=base_reqs,
    # extras_require cho phép người dùng chọn cài thêm tính năng nâng cao
    extras_require={
        "ml": ml_reqs,
        "llm": llm_reqs,
        "all": ml_reqs + llm_reqs  # Cài tất cả
    }
)