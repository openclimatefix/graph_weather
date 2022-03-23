"""Setup"""
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
install_requires = (this_directory / "requirements.txt").read_text().splitlines()
long_description = (this_directory / "README.md").read_text()

setup(
    name="graph_weather",
    version="1.0.5",
    packages=find_packages(),
    url="https://github.com/openclimatefix/graph_weather",
    license="MIT License",
    company="Open Climate Fix Ltd",
    author="Jacob Bieker",
    install_requires=install_requires,
    extras_requires={"vis": ["matplotlib"], "data": ["pysolar"]},
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
)
