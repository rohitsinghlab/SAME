from setuptools import setup, find_packages

setup(
    name="same",
    version="0.1.0",
    description="Spatial Alignment for Multi-modal Experiments",
    author="ap756",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "gurobipy",
        "tqdm",
        "networkx",
        "scikit-learn",
        "alphashape",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
