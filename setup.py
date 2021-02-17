
import pathlib
from setuptools import find_packages, setup

reqs = open("requirements.txt", "r").read().split('\n')
packages = find_packages(exclude=("tests",))

setup(
    name="jai",
    version="0.0.1",
    packages=packages,
    include_package_data=True,
    install_requires=reqs,
)