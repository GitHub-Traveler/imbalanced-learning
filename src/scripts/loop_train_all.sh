#!/bin/bash
resnet=39
vgg=0
for i in `seq 1 $resnet`
do
    python main.py -d 1 --dataset 1 -m 1
done &

for i in `seq 1 $vgg`
do
    python main.py -d 0 --dataset 1 -m 0
done
