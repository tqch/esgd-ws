from setuptools import setup

setup(
    name="esgd-ws",
    version="0.0.9",
    author="Tianqi Chen",
    description="Evolutionary Stochastic Gradient Descent with Weight Sampling",
    license="MIT",
    url="https://github.com/tqch/esgd-ws",
    packages=["models"],
    py_modules=["esgd","esgd-ws"]
)