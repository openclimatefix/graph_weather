"""Setup"""
from pathlib import Path
from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="graph_weather",
    version="1.0.14",
    url="https://github.com/mishooax/graph_weather",
    license="MIT License",
    company="ECMWF / Open Climate Fix Ltd",
    author="Jacob Bieker (original author), with modifications by ECMWF",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email="jacob@openclimatefix.org",
    description="Weather Forecasting with Graph Neural Networks",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "gnn-wb-train=graph_weather.train.wb_train:main",
        ]
    },
)
