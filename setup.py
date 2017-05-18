from setuptools import setup, find_packages

setup(
    name='kaftools',
    version='0.1',
    packages=find_packages(),
    url='',
    license='MIT',
    author='Cristóbal Silva',
    author_email='crsilva@ing.uchile.cl',
    description='Small extensible package for Kernel Adaptive Filtering (KAF) methods.',
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
    ]
)
