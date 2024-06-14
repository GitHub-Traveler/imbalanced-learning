#!/bin/bash
python train_cifar10_noaccel.py -d 0 &
python train_cifar10_noaccel.py -d 1 &
python train_cifar10_noaccel.py -d 2 &
python train_cifar10_noaccel.py -d 3 &
python train_cifar10_noaccel.py -d 4 &
python train_cifar10_noaccel.py -d 5