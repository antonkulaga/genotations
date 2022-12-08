from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.8'
DESCRIPTION = 'Genotations - python library to work with genomes and primers'
LONG_DESCRIPTION = 'Genotations - python library to work with genomes and primers'

# Setting up
setup(
    name="genotations",
    version=VERSION,
    author="antonkulaga (Anton Kualga)",
    author_email="<antonkulaga@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['pyfunctional', 'more-itertools', 'click', 'pycomfort', 'polars', 'genomepy', 'primer3-py', "dna_features_viewer", "pyarrow"],
    keywords=['python', 'utils', 'files'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)