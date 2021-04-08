from setuptools import setup, find_packages


def load_requirements():
    with open("requirements.txt", "r") as f:
        return f.readlines()


setup(
    name="esgd-ws",
    version="0.1.0",
    author="Tianqi Chen",
    description="Evolutionary Stochastic Gradient Descent with Weight Sampling",
    license="MIT",
    url="https://github.com/tqch/esgd-ws",
    package=find_packages(),
    install_requires=load_requirements()
)
