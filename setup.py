#!/usr/bin/env python

try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

from setuptools import find_packages
import os

readme = open("README.rst").read()

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

required_packages = ['numpy>=1.13', 'scipy>=0.14.0', 'configparser',
	'pandas>=0.24.2','corner>=2.0.1','tensorflow-probability>=0.12.0',
	'tensorflow>=2.4.0','matplotlib>=3.3.0','numba>=0.48.0',
	'tqdm>=4.42.1','emcee>=3.0.2','multiprocess>=0.70.10',
	'schwimmbad>=0.3.0']

setup(
	name='ovejero',
	version='1.0.0',
	description='Strong lens Bayesian Neural Network package.',
	long_description=readme,
	author='Sebastian Wagner-Carena',
	author_email='sebaswagner@outlook.com',
	url='https://github.com/swagnercarena/ovejero',
	packages=find_packages(PACKAGE_PATH),
	package_dir={'ovejero': 'ovejero'},
	include_package_data=True,
	install_requires=required_packages,
	license='MIT',
	zip_safe=False
)
