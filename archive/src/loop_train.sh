#!/bin/bash
max=10
for i in `seq 1 $max`
do
    python train_cifar10_noaccel.py -d 1 -i 0.1
done
