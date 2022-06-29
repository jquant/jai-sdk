"""
"""
import os
import re
from setuptools import find_packages, setup

ROOT_DIR = os.path.dirname(__file__)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    dependencies = f.read().splitlines()


def find_version(*filepath):
    # Extract version information from filepath
    with open(os.path.join(ROOT_DIR, *filepath)) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


setup(
    name="jai-sdk",
    version=find_version("jai", "__init__.py"),
    author="JQuant",
    author_email="jedis@jquant.com.br",
    description="JAI - Trust your data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jquant/jai-sdk",
    packages=find_packages(exclude=["tests", "jai.tests"]),
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=dependencies,
)
