# %%
from models import *
from train import *
from tqdm import tqdm

import wandb

import argparse
from torch import optim

classes_cifar10 = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes_mnist = (str(i) for i in range(0, 10))
# Default config (Optimal)

# train_config = {
#     'model_type': 'VGG11',
#     'lr': 0.1,
#     'weight_decay': 1e-5,
#     'batch_size': 128,
#     'momentum': 0.9,
#     'num_epochs': 20,
# }

if __name__ == "__main__":
    # %% Argparse
    msg = "Script for running training on CIFAR-10/MNIST Dataset"
    parser = argparse.ArgumentParser(description = msg)
    parser.add_argument('-d', '--device', help = 'Choose a CUDA device to operate on (0 - num_device). Default: CUDA:0', default= '0' )
    parser.add_argument('-n', '--name', help = 'Name of the runs')
    parser.add_argument('-i', '--imb_degree', type=float, help = 'The degree of imbalancing of specific class in the dataset. Default: 1', default = 1)
    parser.add_argument('--dataset', type=int, choices=[0, 1], 
                        help = 'Choose dataset to perform the training on (0: MNIST, 1: CIFAR10). Default: 1', default = 1)
    parser.add_argument('-m', '--model', type=int, choices=[0, 1], 
                        help = 'Choose model to train (0: VGG11, 1: ResNet18). Default: 0', default = 0)
    args = parser.parse_args()

    # %% Config of the run
    train_config = {
        'dataset': 'CIFAR10' if args.dataset == 1 else 'MNIST',
        'deterministic': False, 
        'model_type': 'ResNet18' if args.model == 1 else 'VGG11',
        'lr': 0.1,
        'weight_decay': 0,
        'batch_size': 128,
        'momentum': 0,
        'num_epochs': 20,
        'imb_degree': args.imb_degree
    }

    device = torch.device('cuda:' + args.device)

    # %% Setup the datasets, dataloader and class weights
    train_dataloader, test_dataloader, class_weights = load_train_test_dataloader(train_config, device)

    # %% Initiate models, criterion, optimizer and scheduler
    if train_config['model_type'] == 'VGG11':
        net = VGG('VGG11')
    elif train_config['model_type'] == 'ResNet18':
        net = ResNet18()
    
    net.to(device)
    criterion = nn.CrossEntropyLoss(weight = class_weights)
    optimizer = optim.SGD(net.parameters(), lr=train_config['lr'],
                        momentum=train_config['momentum'], weight_decay=train_config['weight_decay'])
    
    # %% Record on WANDB.AI
    run = initialize_wandb_run(train_config)
    # %% Training
    train(net, criterion, optimizer, train_config, train_dataloader, test_dataloader, device, run)