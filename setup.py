from setuptools import setup, find_packages

setup(
    name="cube-rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "numpy>=1.26.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.10",
)
