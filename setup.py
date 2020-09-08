#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = ["numpy", "pandas", "xarray>=0.10", "bokeh>=2.2"]

setup(
    author="Peter Hausamann",
    author_email="peter.hausamann@tum.de",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Visualizing xarray data.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="xrview",
    name="xrview",
    packages=find_packages(),
    url="https://github.com/phausamann/xrview",
    version="0.2.0",
    zip_safe=False,
)
