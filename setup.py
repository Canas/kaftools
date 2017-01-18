from setuptools import setup, find_packages

setup(
    name='kernel_filtering',
    version='0.1',
    packages=find_packages(),
    url='',
    license='BSD',
    author='Cristóbal Silva',
    author_email='crsilva@ing.uchile.cl',
    description='Small library with Kernel Adaptive Filtering (KAF) methods.',
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
    ]
)
