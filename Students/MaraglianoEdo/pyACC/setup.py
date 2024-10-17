
from setuptools import setup, find_packages

setup(
    name='pyACC',
    version='0.1.1',
    author='EdoardoMaragliano',
    author_email='edoardo.maragliano@gmail.com',
    description='A short description of your package',
    long_description='A longer description of your package',
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/your-package-name',
    packages=find_packages(),
    install_requires=[
        # Add any dependencies your package requires
        'numpy',
        'matplotlib',
        'camb',
        'pandas',
        'seaborn',
        'scipy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9.18',
)