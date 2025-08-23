from setuptools import setup, find_packages


setup(
    name = "OrderBookAnalytics",
    version = "0.1.0",
    packages = find_packages(),
    install_requires = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "dataclasses>=0.6",
        "typing>=3.7.4",
    ],
    python_requires = ">=3.8",
    author = "Thomas van der Hulst",
    description = "Order book reconstruction and analytics",
)