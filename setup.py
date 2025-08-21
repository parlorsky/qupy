"""
Setup script for the Backtesting Engine.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return requirements

# Package metadata
setup(
    name="backtesting-engine",
    version="1.0.0",
    author="Backtesting Engine Team",
    author_email="contact@backtesting-engine.com",
    description="A comprehensive backtesting framework for quantitative trading strategies",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/backtesting-engine",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/backtesting-engine/issues",
        "Documentation": "https://github.com/your-username/backtesting-engine#readme",
        "Source Code": "https://github.com/your-username/backtesting-engine",
        "Changelog": "https://github.com/your-username/backtesting-engine/blob/main/CHANGELOG.md",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "plotting": ["matplotlib>=3.5.0"],
        "fast": ["polars>=0.18.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "jupyter": ["jupyter>=1.0.0", "ipython>=7.0.0"],
    },
    entry_points={
        "console_scripts": [
            "backtesting-engine=engine.cli:main",  # If we add a CLI later
        ],
    },
    include_package_data=True,
    package_data={
        "engine": ["py.typed"],  # For type checking support
    },
    keywords=[
        "backtesting",
        "trading", 
        "quantitative-finance",
        "strategy-testing",
        "financial-analysis",
        "algorithmic-trading",
        "portfolio-analysis",
        "risk-management"
    ],
    zip_safe=False,  # For better compatibility
)