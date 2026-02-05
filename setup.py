from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="same-spatial",
    version="0.1.0",
    author="Aditya Pratapa",
    author_email="aditya.pratapa@duke.edu",
    description="Spatial Alignment for Multi-modal Experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/singhlab/SAME",
    project_urls={
        "Bug Tracker": "https://github.com/singhlab/SAME/issues",
        "Documentation": "https://singhlab.github.io/SAME/",
    },
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "gurobipy>=10.0.0",
        "tqdm>=4.60.0",
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "geometry": [
            "alphashape>=1.3.0",
            "shapely>=2.0.0",
        ],
        "notebooks": [
            "pooch>=1.6.0",
            "matplotlib>=3.4.0",
            "jupyter",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=9.0.0",
            "mkdocstrings[python]>=0.20.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords="spatial-omics, multimodal, alignment, single-cell, MIP, optimization",
)
