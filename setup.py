import os
from setuptools import setup

setup(
    name = "neuroanalysis",
    version = "0.0.1",
    author = "Luke Campagnola",
    author_email = "lukec@alleninstitute.org",
    description = ("Functions and modular user interface tools for analysis of patch clamp experiment data."),
    license = "MIT",
    keywords = "neuroscience analysis neurodata without borders nwb ",
    url = "http://github.com/aiephys/neuroanalysis",
    packages=['neuroanalysis'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
    ],
)


