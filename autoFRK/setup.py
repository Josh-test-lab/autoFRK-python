"""
Title: Setup file of autoFRK-Python Project
Author: Hsu, Yao-Chih
Version: 1140807
Reference:
"""
from setuptools import setup, find_packages

setup(
    name="autoFRK",
    version="0.1.0",
    description="autoFRK: Automatic Fixed Rank Kriging in Python",
    author="Hsu, Yao-Chih",
    author_email="your.email@example.com",
    url="https://github.com/josh-test-lab/autoFRK-python",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "numpy",
        "faiss-cpu",
        "faiss-gpu",
        "scikit-learn",
    ],
    python_requires=">=3.12",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
