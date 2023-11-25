# Motion Direction Awareness: A Biomimetic Dynamic Capture Mechanism for Video Prediction
![GitHub stars](https://img.shields.io/github/stars/LintureGrant/MDANet)  ![GitHub forks](https://img.shields.io/github/forks/LintureGrant/MDANet?color=green) 

This repository contains the implementation code for the paper:

__Motion Direction Awareness: A Biomimetic Dynamic Capture Mechanism for Video Prediction__

## Introduction

![MDANet](/img/overview.png "The overall framework of MDANet")


MDANet contains an MLP-like temporal module, presenting a new paradigm for efficient video prediction. 

## Overview

* `API/` contains dataloaders and metrics.
* `cls_MD/` contains the implement of MD-Translator.
* `MDAModel.py` contains the MDANet model.
* `run.py` is the executable python file with possible arguments.
* `core.py` is the core file for model training, validating, and testing. 
* `param.py` is the parameter configuration.

## Preparation

### 1. Environment install
We provide the environment requirements file for easy reproduction:
```
  conda create -n MDANet python=3.7
  conda activate MDANet

  pip install -r requirements.txt
```
### 2. Dataset download

Our model has been experimented on the following four datasets:
* [Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/)
* [KTH](https://www.csc.kth.se/cvap/actions/)
* [Human3.6M](http://vision.imar.ro/human3.6m/description.php) 
* [WeatherBench](https://github.com/pangeo-data/WeatherBench)

We provide a download script for the Moving MNIST dataset:

```
  cd ./data/moving_mnist
  bash download_mmnist.sh 
```

### 3. Model traning

This example provide the detail implementation on Moving MNIST, you can easily reproduce our work using the following command:

```
conda activate MDANet
python run.py             
```
Please note that __the model training must strictly adhere to the hyperparameter settings provided in our paper__; otherwise, reproducibility may not be guaranteed.

## Result：

MDANet predicts more accurate actions with less motion blurring compared to other models. Here are some qualitative visualization examples on the KTH dataset:


MDANet


![MDANet](/img/mda_1.gif "MDANet")


SimVP:

![SimVP](/img/simvp_a.gif "SimVP")

PredRNNv2:

![PredRNNv2](/img/predrnnv2_a.gif "PredRNNv2")

PredRNNv1:

![PredRNNv1](/img/predrnnv1_a.gif "PredRNNv1")

ConvLSTM:

![ConvLSTM](/img/convlstm_a.gif "ConvLSTM")