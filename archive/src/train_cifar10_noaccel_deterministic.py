# %%
# Import ML engine
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
import torchvision.datasets as datasets
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from collections import Counter
# Import prebuilt models
from models import *

# Import progress bar
from tqdm import tqdm

# Weights and Biases API
import wandb

import argparse
import numpy as np
import random
# %% Argparse
msg = "Script for running training on CIFAR-10 Dataset"
parser = argparse.ArgumentParser(description = msg)
parser.add_argument('-d', '--device', help = 'Choose a CUDA device to operate on (0 - num_device). Defaut: CUDA:0', default=0)

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
    'deterministic': True,
    'model_type': 'ResNet18',
    'lr': 0.1,
    'weight_decay': 0,
    'batch_size': 128,
    'momentum': 0,
    'num_epochs': 20,
}
device = torch.device('cuda:' + args.device)
generator = torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False


# %% Setup the datasets
transform_func = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True)
])

train_dataset = datasets.CIFAR10(root='../cifar10_dataset', train=True, download=True, transform=transform_func)
test_dataset = datasets.CIFAR10(root='../cifar10_dataset', train=False, download=True, transform=transform_func)

# Specify the class you want to reduce
target_class = 0

# Get indices of all samples of the target class
target_class_indices = [i for i, label in enumerate(train_dataset.targets) if label == target_class]

# Determine how many samples to keep
num_to_keep = len(target_class_indices) // 2  # For example, keep only half

indices_to_keep = np.random.choice(target_class_indices, num_to_keep, replace=False)

# Get indices of all other classes
non_target_class_indices = [i for i in range(len(train_dataset)) if train_dataset.targets[i] != target_class]

# Combine the indices to keep
final_indices = list(indices_to_keep) + non_target_class_indices

# Create a subset of the dataset with the selected indices
subset = Subset(train_dataset, final_indices)

train_dataloader = DataLoader(train_dataset, shuffle = True, num_workers=16, batch_size = train_config['batch_size'], pin_memory=True, prefetch_factor=4, generator = generator)
test_dataloader = DataLoader(test_dataset, shuffle = False, num_workers=16, batch_size = train_config['batch_size'], generator = generator)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# %% Set up class weights
labels = [train_dataset.targets[i] for i in final_indices]
counter = Counter(labels)

class_weights = {x:len(labels)/(10 * counter[x]) for x in range(0, 10)}
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