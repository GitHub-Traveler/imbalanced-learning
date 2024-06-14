# %%
# Import ML engine

# Import prebuilt models
from models import *
from utils import *
from tqdm import tqdm

import wandb

import argparse
import numpy as np
import random
# %% Argparse
msg = "Script for running training on CIFAR-10 Dataset"
parser = argparse.ArgumentParser(description = msg)
parser.add_argument('-d', '--device', help = 'Choose a CUDA device to operate on (0 - num_device). Default: CUDA:0', default=0)
parser.add_argument('-n', '--name', help = 'Name of the runs')
parser.add_argument('-i', '--imb_degree', help = 'The degree of imbalancing of specific class in the dataset. Default: 1', default = 1)
args = parser.parse_args()

# %% Config of the run
# train_config = {
#     'model_type': 'VGG11',
#     'lr': 0.1,
#     'weight_decay': 1e-5,
#     'batch_size': 128,
#     'momentum': 0.9,
#     'num_epochs': 20,
# }

train_config = {
    'dataset': 'CIFAR10',
    'deterministic': False,
    'model_type': 'ResNet18',
    'lr': 0.1,
    'weight_decay': 0,
    'batch_size': 128,
    'momentum': 0,
    'num_epochs': 20,
    'imb_degree': float(args.imb_degree)
}
device = torch.device('cuda:' + args.device)


# %% Setup the datasets

classes_cifar10 = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# %% Set up class weights

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %% Initiate models, criterion, optimizer and scheduler
if train_config['model_type'] == 'VGG11':
    net = VGG('VGG11')
elif train_config['model_type'] == 'ResNet18':
    net = ResNet18()

criterion = nn.CrossEntropyLoss(weight = class_weights)
optimizer = optim.SGD(net.parameters(), lr=train_config['lr'],
                      momentum=train_config['momentum'], weight_decay=train_config['weight_decay'])

# %% Cast to GPU
net.to(device)
# %% Record on WANDB.AI
run = wandb.init(
    # set the wandb project where this run will be logged
    project="cifar10-learning-dynamics",

    # track hyperparameters and run metadata
    config={
    "learning_rate": train_config['lr'],
    "architecture": train_config['model_type'],
    "dataset": "CIFAR-10",
    "epochs": train_config['num_epochs'],
    "deterministic": train_config['deterministic'],
    "imb_degree": train_config['imb_degree']
    },
    name='cifar10_test_run'
)

run.define_metric("epoch")
run.define_metric("train_loss", step_metric="epoch")
run.define_metric("train_accuracy", step_metric="epoch")
run.define_metric("test_loss", step_metric="epoch")
run.define_metric("test_accuracy", step_metric="epoch")
# %% Training
print('Total number of parameters: ', sum([p.numel() for p in net.parameters()]))

for i in tqdm(range(train_config['num_epochs'])):
    net.train()
    train_loss = 0
    correct = 0
    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        label_outputs = outputs.argmax(dim = 1)
        correct += (label_outputs == targets).float().sum()
        run.log({
            'train_loss_step': loss.item()
        })

    net.eval()
    test_loss = 0
    correct_test = 0
    with torch.no_grad():
        for inputs_test, targets_test in test_dataloader:
            inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)
            outputs_test = net(inputs_test)
            correct_test += (outputs_test.argmax(dim = 1) == targets_test).float().sum()
            test_loss += criterion(outputs_test, targets_test)


    train_accuracy = 100 * correct / len(train_dataset)
    test_accuracy = 100 * correct_test / len(test_dataset)
    run.log({
        "epoch": i,
        "train_loss": train_loss, 
        "train_accuracy": train_accuracy, 
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })

wandb.finish()