"""
"""
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    dependencies = f.read().splitlines()

setup(
    name="jai-sdk",
    version="0.11.0",
    author="JQuant",
    author_email="jedis@jquant.com.br",
    description="JAI - Trust your data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jquant/jai-sdk",
    packages=find_packages(exclude=['tests', 'jai.tests']),
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    python_requires=">=3.7",
    install_requires=dependencies,
)
