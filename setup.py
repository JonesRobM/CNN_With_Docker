"""
Setup script for MNIST CNN Classification project
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements from requirements.txt
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

setup(
    name='mnist-cnn',
    version='1.0.0',
    description='MNIST CNN Classification with Docker - A reproducible containerised deep learning workflow',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='MNIST CNN Project',
    python_requires='>=3.8',
    packages=find_packages(exclude=['tests', 'tests.*']),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'matplotlib>=3.7.0',
        'numpy>=1.24.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'scikit-learn>=1.3.0',
            'seaborn>=0.12.0',
        ],
        'test': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
        ],
        'viz': [
            'scikit-learn>=1.3.0',
            'seaborn>=0.12.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'mnist-train=train_cnn:main',
            'mnist-predict=predict_cnn:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    keywords='mnist cnn deep-learning pytorch docker machine-learning',
    project_urls={
        'Source': 'https://github.com/yourusername/mnist-cnn-docker',
        'Issues': 'https://github.com/yourusername/mnist-cnn-docker/issues',
    },
    include_package_data=True,
    zip_safe=False,
)
