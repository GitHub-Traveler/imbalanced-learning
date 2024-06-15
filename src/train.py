"""
Module containing necessary functions for loading, training and evaluating the model
"""

import torch
import torchvision.transforms.v2 as v2
import torchvision.datasets as datasets

from torch.utils.data import DataLoader, Subset
from collections import Counter
import numpy as np
import wandb
from tqdm import tqdm

__all__ = ['load_train_test_dataloader', 'initialize_wandb_run', 'train']
def load_train_test_dataloader(train_config, device):
    transform_func = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
    ])
    if train_config['dataset'] == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='/home/ubuntu/21thanh.ht/imbalanced_dynamics/data/cifar10_dataset', train=True, download=True, transform=transform_func)
        test_dataset = datasets.CIFAR10(root='/home/ubuntu/21thanh.ht/imbalanced_dynamics/data/cifar10_dataset', train=False, download=True, transform=transform_func)
    elif train_config['dataset'] == 'MNIST':
        train_dataset = datasets.MNIST(root='/home/ubuntu/21thanh.ht/imbalanced_dynamics/data/cifar10_dataset', train=True, download=True, transform=transform_func)
        test_dataset = datasets.MNIST(root='/home/ubuntu/21thanh.ht/imbalanced_dynamics/data/cifar10_dataset', train=False, download=True, transform=transform_func)
    # Specify the class you want to reduce
    target_class = 0

    # Get indices of all samples of the target class
    target_class_indices = [i for i, label in enumerate(train_dataset.targets) if label == target_class]

    # Determine how many samples to keep
    num_to_keep = int(len(target_class_indices) * float(train_config['imb_degree']))  

    indices_to_keep = np.random.choice(target_class_indices, num_to_keep, replace=False)

    # Get indices of all other classes
    non_target_class_indices = [i for i in range(len(train_dataset)) if train_dataset.targets[i] != target_class]

    # Combine the indices to keep
    final_indices = list(indices_to_keep) + non_target_class_indices

    # Create a subset of the dataset with the selected indices
    subset = Subset(train_dataset, final_indices)

    train_dataloader = DataLoader(train_dataset, shuffle = True, num_workers=16, batch_size = train_config['batch_size'], pin_memory=True, prefetch_factor=4)
    test_dataloader = DataLoader(test_dataset, shuffle = False, num_workers=16, batch_size = train_config['batch_size'])
    labels = [train_dataset.targets[i] for i in final_indices]
    counter = Counter(labels)

    class_weights = {x:len(labels)/(10 * counter[x]) for x in range(0, 10)}
    class_weights = torch.Tensor(list(class_weights.values())).to(device)
    return train_dataloader, test_dataloader, class_weights

def initialize_wandb_run(train_config):
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="cifar10-learning-dynamics",

        # track hyperparameters and run metadata
        config={
        "dataset": train_config['dataset'],
        "learning_rate": train_config['lr'],
        "architecture": train_config['model_type'],
        "dataset": "CIFAR-10",
        "epochs": train_config['num_epochs'],
        "deterministic": train_config['deterministic'],
        "imb_degree": train_config['imb_degree']
        },
        name = f"{train_config['dataset']}_{train_config['model_type']}_run"
    )

    run.define_metric("epoch")
    run.define_metric("train_loss", step_metric="epoch")
    run.define_metric("train_accuracy", step_metric="epoch")
    run.define_metric("test_loss", step_metric="epoch")
    run.define_metric("test_accuracy", step_metric="epoch")
    return run

def train(net, criterion, optimizer, train_config, train_dataloader, test_dataloader, device, wandb_run):
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
            wandb_run.log({
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


        train_accuracy = 100 * correct / len(train_dataloader.dataset)
        test_accuracy = 100 * correct_test / len(test_dataloader.dataset)
        wandb_run.log({
            "epoch": i,
            "train_loss": train_loss, 
            "train_accuracy": train_accuracy, 
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

    wandb_run.finish()