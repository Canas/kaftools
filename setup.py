import os
from codecs import open
from setuptools import setup, find_packages


# Get long description from README
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='kaftools',
    version='0.1.1',
    description='Small extensible package for Kernel Adaptive Filtering (KAF) methods.',
    long_description=long_description,
    url='https://github.com/canas/kaftools',
    author='Cristóbal Silva',
    author_email='crsilva@ing.uchile.cl',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='kernel adaptive filters kaf'
)
