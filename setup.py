from setuptools import setup, find_packages

setup(
    name="unilm",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        "transformers",
        "torch",
    ],
    entry_points={
        "console_scripts": [
            "unilm_train=unilm.procedures:run_train",
            "unilm_decode=unilm.procedures:run_decode",
        ]
    }
)
