#!/bin/bash

CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_cifar10_noaccel_deterministic.py -d 0 &
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_cifar10_noaccel_deterministic.py -d 1 &
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_cifar10_noaccel_deterministic.py -d 2 &
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_cifar10_noaccel_deterministic.py -d 3 &
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_cifar10_noaccel_deterministic.py -d 4 &
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_cifar10_noaccel_deterministic.py -d 5