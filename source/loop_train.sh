#!/bin/bash
max=100
for i in `seq 1 $max`
do
    python train_cifar10_noaccel.py -d 0 -i 1
done
