# TODO: Adding training and collecting loops for MNIST, as well as VGG-11 Model.
# %%
from models import *
from train import *
from metrics import *
import json
import os
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
    # parser.add_argument('-n', '--name', help = 'Name of the runs')
    parser.add_argument('-i', '--imb-degree', type=float, help = 'The degree of imbalancing of half of the classes in the dataset. Default: 1', default = 1)
    parser.add_argument('--dataset', type=int, choices=[0, 1], 
                        help = 'Choose dataset to perform the training on (0: MNIST, 1: CIFAR10). Default: 1', default = 1)
    parser.add_argument('--class-weights', action = 'store_true',
                        help = 'Adding class weights inversely porpotional to the dataset size in training. Default = False', default = False)
    parser.add_argument('-m', '--model', type=int, choices=[0, 1], 
                        help = 'Choose model to train (0: VGG11, 1: ResNet18). Default: 0', default = 0)
    parser.add_argument('--n-runs', type=int, help = 'Number of training loops. Default: 1', default = 1)
    parser.add_argument('--output-dir', help='Path for the output of the collected runs', default = None)
    parser.add_argument('--eval-every', type=int, help='Number of steps taken between each two evaluation process', default = 40)
    parser.add_argument('--n-epochs', type=int, help = 'Number of epoches each training loops. Default: 20', default = 20)
    parser.add_argument('--imbalance-loss', action = 'store_true',
                        help = 'Recording the accuracy of minority and majority classes separately', default = False)
    args = parser.parse_args()

    # %% Config of the run
    train_config = {
        'dataset': 'CIFAR10' if args.dataset == 1 else 'MNIST',
        'model_type': 'ResNet18' if args.model == 1 else 'VGG11',
        'lr': 0.001,
        'weight_decay': 0,
        'batch_size': 128,
        'momentum': 0,
        'num_epochs': args.n_epochs,
        'weighted_training': args.class_weights,
        'n_runs': args.n_runs,
        'output_dir': args.output_dir,
        'eval_every': args.eval_every,
        'imb_degree': [args.imb_degree, args.imb_degree, args.imb_degree, args.imb_degree, args.imb_degree, 1, 1, 1, 1, 1],
        'imbalance_loss': args.imbalance_loss
    }

    device = torch.device('cuda:' + args.device)
    for run in range(train_config['n_runs']):
        # %% Setup the datasets, dataloader and class weights
        train_dataloader, test_dataloader, class_weights, val_dataloaders = load_train_test_dataloader(train_config, device)
        # %% Initiate models, criterion, optimizer and scheduler
        if train_config['model_type'] == 'VGG11':
            net = VGG('VGG11')
        elif train_config['model_type'] == 'ResNet18':
            net = ResNet18()
        
        net.to(device)
        # eval_criterion = nn.CrossEntropyLoss()
        eval_criterion = nn.CrossEntropyLoss(reduction='none')
        if train_config['weighted_training']:
            criterion = nn.CrossEntropyLoss(weight = class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=train_config['lr'],
                            momentum=train_config['momentum'], weight_decay=train_config['weight_decay'])
        # %% Record on WANDB.AI
        run_output_dir = os.path.join(train_config['output_dir'], f"run{run}")
        wandb_run = initialize_wandb_run(train_config)
        if not os.path.exists(run_output_dir):
            os.makedirs(run_output_dir)
        # %% Training
        train(net, criterion, eval_criterion, optimizer, train_config, train_dataloader, test_dataloader, device, wandb_run, run_output_dir, val_dataloaders)