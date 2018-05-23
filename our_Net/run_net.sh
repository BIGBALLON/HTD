#!/usr/bin/env bash
Datasets=(cifar10 cifar100 fashion_mnist)
# schedulers:
#  0 step_decay
#  1 exponential
#  2 two_stage_exponential
#  3 tanh_restart
#  4 cos_restart
#  5 tanh
#  6 cos
#  7 cos_iteration
#  8 tanh_iteration
LRScheduler=step_decay # one of schedulers
Depth=5 # depth of ResNet and Wide ResNet
Width=1 # width of Wide ResNet
BatchSize=128
Epochs=200
Net=resnet
# Net = lenet or resnet or wresnet
TanhBegin=-4.0 # L of [L, U]
TanhEnd=4.0    # U of [L, U]
let depth=3*2*Depth+2

# LeNet
# python3 train.py " -b $BatchSize -e $Epochs -d ${Datasets[$k]} -lr_m ${LRScheduler} -net ${Net} " 

# ResNet-d
# python3 train.py " -b $BatchSize -e $Epochs -d ${Datasets[$k]} -lr_m ${LRScheduler} -net ${Net} -depth $Depth " 

# WRN-d-w
# python3 train.py " -b $BatchSize -e $Epochs -d ${Datasets[$k]} -lr_m ${LRScheduler} -net ${Net} -depth $Depth -width $Width "

# if using tanh for ResNet-d
# python3 train.py " -b $BatchSize -e $Epochs -d ${Datasets[$k]} -lr_m ${LRScheduler} -net ${Net} -depth $Depth -tanh_begin ${TanhBegin} -tanh_end ${TanhEnd}" 
