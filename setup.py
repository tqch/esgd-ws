from setuptools import setup, find_packages


def load_requirements():
    with open("requirements.txt", "r") as f:
        return f.read().split("\n")


setup(
    name="esgd-ws",
    version="1.0.1",
    author="Tianqi Chen",
    description="Evolutionary Stochastic Gradient Descent with Weight Sampling",
    license="MIT",
    url="https://github.com/tqch/esgd-ws",
    packages=find_packages(),
    install_requires=load_requirements()
)
