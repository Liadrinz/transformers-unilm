from setuptools import setup, find_packages

setup(
    name="unilm",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        "transformers",
        "torch",
    ],
)
