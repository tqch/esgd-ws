# Evolutionary Stochastic Gradient Descent with Weight Sampling

## Table of Contents

- [Installation](## Installation)
- [Usage](## Usage)

This repo provides a PyTorch Implementation of a modification to *Evolutionary Stochastic Gradient Descent* (**ESGD**) [[arxiv]](https://arxiv.org/abs/1810.06773), which we named **ESGD-WS**.

## Installation

```
pip install "git+https://github.com/tqch/esgd-ws#egg=wsgd-ws"
```

## Usage
```
python -m esgd-ws [-a] [--dataset DATASET] {baseline|esgd|esgd_ws}
```

