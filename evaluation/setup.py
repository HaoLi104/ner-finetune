from setuptools import setup, find_packages

setup(
    name="evaluation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pytest>=7.0.0",
        "requests>=2.28.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
    ],
    author="Ocean",
    author_email="ocean@nexusera.com",
    description="A Python project to evaluate the performance of medical entity extraction models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
) 