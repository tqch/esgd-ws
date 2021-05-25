# Evolutionary Stochastic Gradient Descent with Weight Sampling

This repo provides an unofficial PyTorch Implementation of *Evolutionary Stochastic Gradient Descent* (**ESGD**) [[arxiv]](https://arxiv.org/abs/1810.06773), and a variant named **ESGD-WS** that uses the weight sampling strategy.

## Table of Contents

- [Installation](##Installation)
- [Usage](##Usage)

## Installation

```
pip install "git+https://github.com/tqch/esgd-ws#egg=esgd-ws"
```

## Usage
```
python -m esgd-ws [-a] [--dataset DATASET] {baseline|esgd|esgd_ws}
```

