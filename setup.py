"""
"""
import pathlib
from setuptools import find_packages, setup


setup(
    name="mycelia-sdk",
    version="0.1.1",
    author="rogerio.guicampos",
    author_email="rogerio.campos@jquant.com.br",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=["azure-storage-blob", "brain-plasma", "numpy", "pandas"],
)