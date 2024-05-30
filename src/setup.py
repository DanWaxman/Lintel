from setuptools import setup, find_packages

setup(
    name="Lintel",
    version="0.0.1",
    url="https://github.com/DanWaxman/Lintel",
    author="Dan Waxman",
    author_email="daniel.waxman@stonybrook.edu",
    description="Code for our paper Online Prediction of Switching Gaussian Process Time Series with Constant-Time Updates",
    packages=find_packages(),
    install_requires=[
        "jax == 0.4.4",
        "jaxlib == 0.4.4",
        "jaxtyping",
        "objax == 1.7.0",
        "tensorflow-probability >= 0.22.1",
        "tqdm >= 4.66.1",
        "bayesnewton == 1.3.4",
        "scipy == 1.12.0",
        "numpy == 1.26.4",
    ],
)
