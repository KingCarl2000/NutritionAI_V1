"""
The setup.py file is an essential part of packaging and 
distributing Python projects.
"""

from setuptools import find_packages, setup
from typing import List
import os

def get_requirements(file_path: str) -> List[str]:
    """
    This function returns a list of valid requirements from a specified text file.
    """
    requirement_lst: List[str] = []
    
    if not os.path.exists(file_path):
        print(f"Requirement file not found at: {file_path}")
        return requirement_lst

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            requirement = line.strip()
            # Bỏ qua dòng trống, comment và các cờ pip (-e ., -r, v.v.)
            if requirement and not requirement.startswith("#") and not requirement.startswith("-"):
                requirement_lst.append(requirement)

    return requirement_lst

# Đọc danh sách từ từng file
base_reqs = get_requirements("requirements/base.txt")
ml_reqs = get_requirements("requirements/ml.txt")
llm_reqs = get_requirements("requirements/llm.txt")

# Tạo dictionary extras_require chỉ với các danh sách không rỗng
extras = {}
if ml_reqs:
    extras["ml"] = ml_reqs
if llm_reqs:
    extras["llm"] = llm_reqs
if ml_reqs or llm_reqs:
    extras["all"] = ml_reqs + llm_reqs

setup(
    name="NutritionAI_V1",
    version="0.0.1",
    author="Carl",
    author_email="ngvcuong282@gmail.com",
    packages=find_packages(),
    install_requires=base_reqs,
    extras_require=extras if extras else None
)