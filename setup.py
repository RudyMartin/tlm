#!/usr/bin/env python3
"""
TLM - Core ML Algorithms (Zero Dependencies)
==========================================
"""

from setuptools import setup, find_packages

# Read version from pyproject.toml
def get_version():
    try:
        import toml
        with open('pyproject.toml', 'r') as f:
            data = toml.load(f)
            return data['project']['version']
    except:
        return '1.2.0'

# Read README if available
def get_long_description():
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return "TLM - Teaching library: NumPy-like API on pure Python lists + Data Science algos (NumPy-free)"

setup(
    name="tlm",
    version=get_version(),
    author="Rudy Martin",
    author_email="rudy@nextshiftconsulting.com",
    description="Teaching library: NumPy-like API on pure Python lists + Data Science algos (NumPy-free)",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/rudymartin/tlm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.9",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            # Future CLI commands can go here
        ],
    },
    zip_safe=False,
    include_package_data=True,
)