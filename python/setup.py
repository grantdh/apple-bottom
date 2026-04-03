"""Setup configuration for apple_bottom Python wrapper."""

from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent.resolve()

# Read the README
readme_file = here / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="apple-bottom",
    version="1.2.0",
    description="FP64-class BLAS library for Apple Silicon GPUs — Python wrapper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/apple-bottom",
    author="Grant Heileman",
    author_email="example@example.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: MacOS",
    ],
    keywords="blas gpu matrix apple-silicon accelerate fp64",
    packages=find_packages(),
    package_data={
        "apple_bottom": ["py.typed"],
    },
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
)
