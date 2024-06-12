import argparse

msg = "Script for running training on CIFAR-10 Dataset"
parser = argparse.ArgumentParser(description = msg)
parser.add_argument('-d', '--device', help = 'Choose a CUDA device to operate on (0 - num_device). Defaut: CUDA:0')

args = parser.parse_args()
print(type(args.device))