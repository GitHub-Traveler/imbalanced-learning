#!/bin/bash
max=34
for i in `seq 1 $max`
do
    python train_cifar10_noaccel.py -d 1 -i 0.1 &
    python train_cifar10_noaccel.py -d 2 -i 0.1 &
    python train_cifar10_noaccel.py -d 5 -i 0.1
done
