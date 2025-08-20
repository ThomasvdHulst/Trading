from setuptools import setup, find_packages

setup(
    name="AdvMarketMakerSimulation",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "arch>=5.3.0",
        "pydantic>=2.0.0",
        "PyYAML>=6.0",
        "matplotlib>=3.4.0",
    ],
    python_requires=">=3.8",
    author="Thomas van der Hulst",
    description="Advanced Market Maker Simulation",
)