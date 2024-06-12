# %%
# Import ML engine
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
import torchvision.datasets as datasets
import torch.optim as optim
import torch.backends.cudnn as cudnn

# Import prebuilt models
from models import *

# Import accelerator
from accelerate import Accelerator
# Import progress bar
from tqdm import tqdm

# Weights and Biases API
import wandb

import argparse
# %% Argparse
msg = "Script for running training on CIFAR-10 Dataset"
parser = argparse.ArgumentParser(description = msg)
parser.add_argument('-d', '--device', help = 'Choose a CUDA device to operate on (0 - num_device). Defaut: CUDA:0', default=0)
parser.add_argument('-a', '--accelerate', action='store_true', help = 'Activate multi-GPU training')
parser.add_argument('-n', '--name', help = 'Name of the runs')
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
    'model_type': 'VGG11',
    'lr': 0.1,
    'weight_decay': 0,
    'batch_size': 128,
    'momentum': 0,
    'num_epochs': 20,
}

# %% Setup the datasets
transform_func = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True)
])

train_dataset = datasets.CIFAR10(root='../cifar10_dataset', train=True, download=True, transform=transform_func)
test_dataset = datasets.CIFAR10(root='../cifar10_dataset', train=False, download=True, transform=transform_func)
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle = True, num_workers=16, batch_size = train_config['batch_size'], pin_memory=True, prefetch_factor=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle = False, num_workers=16, batch_size = train_config['batch_size'])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %% Initiate models, criterion, optimizer and scheduler
if train_config['model_type'] == 'VGG11':
    net = VGG('VGG11')
elif train_config['model_type'] == 'Resnet18':
    net = ResNet18()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=train_config['lr'],
                      momentum=train_config['momentum'], weight_decay=train_config['weight_decay'])

# %% Accelerator
accelerator = Accelerator(split_batches=True)
device = accelerator.device
net, optimizer, train_dataloader = accelerator.prepare(net, optimizer, train_dataloader)
# %% Record on WANDB.AI
if accelerator.is_main_process:
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="cifar10-learning-dynamics",

        # track hyperparameters and run metadata
        config={
        "learning_rate": train_config['lr'],
        "architecture": train_config['model_type'],
        "dataset": "CIFAR-10",
        "epochs": train_config['num_epochs'],
        },
        name=args.name
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
        accelerator.backward(loss)
        optimizer.step()

        train_loss += loss.item()

        label_outputs = outputs.argmax(dim = 1)
        correct += (label_outputs == targets).float().sum()
        if accelerator.is_main_process:
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
    if accelerator.is_main_process:
        run.log({
            "epoch": i,
            "train_loss": train_loss, 
            "train_accuracy": train_accuracy, 
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

wandb.finish()