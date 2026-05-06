# Bridging Inter-View and Client Heterogeneity: Federated Multi-View Clustering under Non-IID Data

This project uses **Python 3.9**, and its dependencies are mainly managed through **pip + conda environment**. This document explains how to generate and correctly install `requirements.txt` based on the existing environment。

---

## 1. Python  and CUDA version

```text
Python 3.9.23
CUDA 11.8
```

## 2. Installation instructions

First, install PyTorch separately:

```
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining dependencies:

```
pip install -r requirements.txt
```

## 3. Operation Instructions

First, please unzip the file "data.zip".

 main.py uses the MNIST-USPS dataset by default. If you want to run the entire dataset at once, please run run.py.