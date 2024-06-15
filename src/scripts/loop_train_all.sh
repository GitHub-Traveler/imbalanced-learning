#!/bin/bash
max=100

for i in `seq 1 $max`
do
    python main.py -d 3 -i 0.1 --dataset 1 -m 1
done &

for i in `seq 1 $max`
do
    python main.py -d 2 -i 0.1 --dataset 1 -m 0
done
