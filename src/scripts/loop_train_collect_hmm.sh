#!/bin/bash

python main.py --n-runs 40 -m 1 --output-dir "../test_0.01/json/resnet18_0.1imb" --eval-every 50 --n-epochs 30 -i 0.1 -d 1 &

python main.py --n-runs 40 -m 1 --output-dir "../results_imb/json/resnet18_0.1imb_weighted" --eval-every 50 --n-epochs 30 -i 0.1 -d 2 --class-weights &

python main.py --n-runs 40 -m 1 --output-dir "../results_imb/json/resnet18_0.5imb" --eval-every 50 --n-epochs 30 -i 0.5 -d 3 &

python main.py --n-runs 40 -m 1 --output-dir "../results_imb/json/resnet18_0.5imb_weighted" --eval-every 50 --n-epochs 30 -i 0.5 -d 4 --class-weights &

python main.py --n-runs 40 -m 1 --output-dir "../results_imb/json/resnet18_balance" --eval-every 50 --n-epochs 30 -d 5


# python main.py --n-runs 1 -m 1 --output-dir "../test" --eval-every 50 --n-epochs 30 -d 2
