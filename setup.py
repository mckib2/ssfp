'''Setup.py'''

from distutils.core import setup
from setuptools import find_packages

setup(
    name='ssfp',
    version='0.4.0',
    author='Nicholas McKibben',
    author_email='nicholas.bgp@gmail.com',
    packages=find_packages(),
    scripts=[],
    url='https://github.com/mckib2/ssfp',
    license='GPLv3',
    description='SSFP simulation',
    long_description=open('README.rst').read(),
    install_requires=[
        "numpy>=1.18.1",
        "matplotlib>=3.1.2",
        "scikit-image>=0.16.2",
        "phantominator>=0.6.1",
        "tqdm>=4.42.0"
    ],
    python_requires='>=3.6',
)
