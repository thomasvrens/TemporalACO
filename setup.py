from setuptools import setup, find_packages

setup(
    name='temporal_aco',
    version='0.3.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.0',
        'matplotlib>=3.8.0',
        'tqdm>=4.65',
        'optuna>=4.0.0'
    ],
    author='Thomas van Rens',
    description='ACO for ETV scheduling',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)