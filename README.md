# Stochastic Gradient Descent with Hyperbolic-Tangent Decay on Classification

This repository contains the code for HTD introduced in the following paper:

[Stochastic Gradient Descent with Hyperbolic-Tangent Decay on Classification](https://arxiv.org/abs/1806.01593) (Accepted to WACV 2019)

## Contents
1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results on CIFAR](#results-on-cifar)
4. [Results on ImageNet](#results-on-imagenet)
5. [Contact](#contact)

## Introduction

Learning rate scheduler has been a critical issue in the deep neural network training. Several schedulers and methods have been proposed, including step decay scheduler, adaptive method, cosine scheduler and cyclical scheduler. This paper proposes a new scheduling method, named hyperbolic-tangent decay (HTD). We run experiments on several benchmarks such as: ResNet, Wide ResNet and DenseNet for CIFAR-10 and CIFAR-100 datasets, LSTM for PAMAP2 dataset, ResNet on ImageNet and Fashion-MNIST datasets. In our experiments, HTD outperforms step decay and cosine scheduler in nearly all cases, while requiring less hyperparameters than step decay, and more flexible than cosine scheduler. 

![htd](https://user-images.githubusercontent.com/7837172/40904487-7967b488-680d-11e8-8339-bcda89d9e8e1.png)


## Usage 

1. (If you want to train CIFAR datasets only) Install tensorflow and keras.
2. (If you want to train ImageNet) Install Torch and required dependencies like cuDNN. 
See the instructions [here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md) for a step-by-step guide.
3. Clone this repo: ```https://github.com/BIGBALLON/HTD.git```

```
├─ our_Net                   % Our CIFAR dataset training code
├─ fb.resnet.torch           % [facebook/fb.resnet.torch]  
└─ DenseNet                  % [liuzhuang13/DenseNet]   
```

See the following examples. To run the training with ResNet, on **CIFAR-10**,   
using **step decay** scheduler, simply run:

```
python train.py --batch_size 128 \
                --epochs 200 \
                --data_set cifar10 \
                --learning_rate_method step_decay \
                --network resnet \
                --log_path ./logs \
                --network_depth 5 
``` 

using **other learning rate scheduler(cos or tanh)**, by changing ``--learning_rate_method`` flag:

```
python train.py --batch_size 128 \
                --epochs 200 \
                --data_set cifar10 \
                --learning_rate_method tanh \
                --network resnet \
                --log_path ./logs \
                --network_depth 5 \
                --tanh_begin -4.0 \
                --tanh_end 4.0
``` 



## Results on CIFAR
The table below shows the results of HTD on CIFAR datasets. Best results are written in blue.   
The character * indicates results are directly obtained from the original paper.

![result_cifar](https://user-images.githubusercontent.com/7837172/40903675-c143afd0-680a-11e8-855b-5cfd5e1eda87.png)

## Results on ImageNet

The Torch models are trained under the same setting as in [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch). Best results are written in blue.   
The character * indicates results are directly obtained from the original paper. 

![result_imagenet](https://user-images.githubusercontent.com/7837172/40903677-c16b20ce-680a-11e8-95ef-6cf674d3a585.png)


## Contact

fm.bigballon at gmail.com  
byshiue at gmail.com   

If you use our code, please consider citing the technical report as follows:

```
@inproceedings{hsueh2019stochastic,
  title={Stochastic Gradient Descent with Hyperbolic-Tangent Decay on Classification},
  author={Hsueh, Bo-Yang and Li, Wei and Wu, I-Chen},
  booktitle={2019 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  pages={435--442},
  year={2019},
  organization={IEEE}
}
```

Please feel free to contact us if you have any discussions, suggestions or questions!!

