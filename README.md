# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains an implementation of Conformer for solving ASR task with PyTorch. It contains training script for end-to-end speech recognition. Wandb is used to for logging. 

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

3. In order to avoid any problems with audio [ffmpeg](https://www.ffmpeg.org/) is recommended to install.

4. The project is exploiting MLOps platforms: [CometML](https://www.comet.com/) or [Wandb](https://wandb.ai/site/). 

You should create your account, get api_key and create variable COMETML_API_KEY or WANDB_API_KEY to be able to run train.py.

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

## Credits / Acknowledgments

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template)

Some optimization tricks are taken from Andrej Karpathy [nanoGPT repo](https://github.com/karpathy/nanoGPT/tree/master)

Part of the code for Conformer is taken from [this repo](https://github.com/jreremy/conformer/tree/master)

The conformer architecture is based on paper: ["Conformer: Convolution-augmented Transformer for Speech Recognition" Google Inc. 2020](https://arxiv.org/pdf/2005.08100)

[FlashAttention paper](https://arxiv.org/pdf/2205.14135)

## Authors:
Yurakhno Konstantin - personal project

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
