"""
Module containing necessary functions for loading, training and evaluating the model
"""

import torch
import torchvision.transforms.v2 as v2
import torchvision.datasets as datasets
import numpy as np
import wandb
import os
import json

from torch.utils.data import DataLoader, Subset
from collections import Counter
from tqdm import tqdm
from metrics import *
# TODO: Adding training and collecting loops for MNIST, as well as VGG-11 Model.
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
 
    fractions = train_config['imb_degree']
    num_classes = len(fractions)
    assert num_classes == 10, "The list of fractions must have 10 elements."

    # Create a list to store the indices to keep for each class
    indices_to_keep = []

    # Iterate over each class
    for target_class in range(num_classes):
        # Get indices of all samples of the current class
        target_class_indices = [i for i, label in enumerate(train_dataset.targets) if label == target_class]

        # Determine how many samples to keep
        num_to_keep = int(len(target_class_indices) * fractions[target_class])

        # Randomly select the indices to keep
        if num_to_keep > 0:
            selected_indices = np.random.choice(target_class_indices, num_to_keep, replace=False)
            indices_to_keep.extend(selected_indices)

    # Create a subset of the dataset with the selected indices
    subset = Subset(train_dataset, indices_to_keep)

    train_dataloader = DataLoader(subset, shuffle = True, num_workers=16, batch_size = train_config['batch_size'], pin_memory=True, prefetch_factor=4)
    test_val_dataloader = DataLoader(test_dataset, shuffle = False, num_workers=16, batch_size = train_config['batch_size'] * 8)
    train_val_dataloader = DataLoader(subset, shuffle = False, num_workers=16, batch_size = train_config['batch_size'] * 8)
    minority = (0, 1, 2, 3, 4)
    majority = (5, 6, 7, 8, 9)
    val_dataloaders = {
        'train': train_val_dataloader,
        'test': test_val_dataloader
    }

    class_weights = [1/fractions[x] for x in range(0, 10)]
    class_weights = torch.Tensor(class_weights).to(device)

    return train_dataloader, test_val_dataloader, class_weights, val_dataloaders


def initialize_wandb_run(train_config):
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="imbalanced-latent-model",

        # track hyperparameters and run metadata
        config={
        "dataset": train_config['dataset'],
        "learning_rate": train_config['lr'],
        "architecture": train_config['model_type'],
        "dataset": "CIFAR-10",
        "epochs": train_config['num_epochs'],
        "imb_degree": train_config['imb_degree']
        },
        name = f"{train_config['dataset']}_{train_config['model_type']}_run"
    )

    run.define_metric("epoch")
    run.define_metric("train_loss", step_metric="epoch")
    run.define_metric("train_accuracy", step_metric="epoch")
    # run.define_metric("eval_loss", step_metric="epoch")
    # run.define_metric("eval_accuracy", step_metric="epoch")
    return run

def train(net, criterion, eval_criterion, optimizer, train_config, train_dataloader, test_dataloader, device, wandb_run, run_output_dir, val_dataloaders):
    print('Total number of parameters: ', sum([p.numel() for p in net.parameters()]))
    step = 0
    minor_classes = [0, 1, 2, 3, 4]
    major_classes = [5, 6, 7, 8, 9]
    for i in tqdm(range(train_config['num_epochs'])):
        train_loss = 0
        correct = 0
        current_loss = 0
        num_batch = 0
        num_samples = 0
        for inputs, targets in train_dataloader:
            net.train()
            inputs, targets = inputs.to(device), targets.to(device)               
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            num_batch += 1
            num_samples += targets.size(0)

            current_loss = loss.detach().cpu().item()
            train_loss += current_loss
            label_outputs = outputs.argmax(dim = 1)
            correct += (label_outputs == targets).int().sum()

            wandb_run.log({
                'train_loss_step': current_loss
            })

            if step % train_config['eval_every'] == 0:
                net.eval()
                with torch.no_grad():
                    # eval_loss, eval_accuracy = eval(net, test_dataloader, eval_criterion, device)
                    # dct = eval_dataloaders(net, val_dataloaders, criterion, device)
                    eval_minority_loss, eval_majority_loss, eval_loss = calculate_average_loss(val_dataloaders['test'], net, eval_criterion, 
                                                                                               minor_classes, major_classes, device)
                    train_minority_loss, train_majority_loss, train_loss = calculate_average_loss(val_dataloaders['train'], net, eval_criterion, 
                                                                                               minor_classes, major_classes, device)
                    print(eval_minority_loss, eval_majority_loss, eval_loss, train_minority_loss, train_majority_loss, train_loss )
                    data = get_metrics_resnet18(net)
                    data['train_loss'] = train_loss
                    data['eval_loss'] = eval_loss
                    data['train_minority_loss'] = train_minority_loss
                    data['train_majority_loss'] = train_majority_loss
                    data['eval_minority_loss'] = eval_minority_loss
                    data['eval_majority_loss'] = eval_majority_loss
                    # data['train_accuracy'] = (100 * correct / num_samples).item()
                    # data['eval_accuracy'] = eval_accuracy.item()
                    # for key, value in dct.items():
                    #     data[key] = value
                
                with open(os.path.join(run_output_dir, f"step{step}.json"), "w") as f:
                    json.dump(data, f)

        train_accuracy = 100 * correct / len(train_dataloader.dataset)
        train_loss = train_loss/num_batch
        wandb_run.log({
            "epoch": i,
            "train_loss": train_loss, 
            "train_accuracy": train_accuracy, 
            # "eval_loss": eval_loss,
            # "eval_accuracy": eval_accuracy
        })

    wandb_run.finish()
    return None

@torch.no_grad()
def eval(net, dataloader, criterion, device):
    total_loss = 0
    batch_count = 0
    correct = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.detach().cpu().item()
        correct += (outputs.argmax(dim = 1) == targets).int().sum()
        batch_count += 1
    
    return total_loss/batch_count, 100 * correct / len(dataloader.dataset)

@torch.no_grad()
def eval_dataloaders(net, val_dataloaders, criterion, device):
    dct = {}
    for name, dataloader in val_dataloaders.items():
        total_loss = 0
        num_batch = 0
        for inputs, targets in dataloader:
            inputs, targets = next(iter(dataloader))
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batch += 1
        dct[name] = total_loss / num_batch
    
    return dct

@torch.no_grad()
def calculate_average_loss(dataloader, model, loss_fn, minority_class_labels, majority_class_labels, device):
    model.eval()  # Set the model to evaluation mode
    
    minority_loss = 0.0
    majority_loss = 0.0
    minority_count = 0
    majority_count = 0
    
    with torch.no_grad():  # No need to track gradients for evaluation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            # Create masks for minority and majority classes
            minority_mask = torch.zeros_like(labels, dtype=torch.bool)
            majority_mask = torch.zeros_like(labels, dtype=torch.bool)
            
            for minority_label in minority_class_labels:
                minority_mask |= (labels == minority_label)
            
            for majority_label in majority_class_labels:
                majority_mask |= (labels == majority_label)
            
            if minority_mask.sum() > 0:
                minority_loss += loss[minority_mask].sum().item()
                minority_count += minority_mask.sum().item()
                
            if majority_mask.sum() > 0:
                majority_loss += loss[majority_mask].sum().item()
                majority_count += majority_mask.sum().item()
    
    average_minority_loss = minority_loss / minority_count if minority_count > 0 else 0.0
    average_majority_loss = majority_loss / majority_count if majority_count > 0 else 0.0
    average_loss = (minority_loss + majority_loss) / (minority_count + majority_count)
    return average_minority_loss, average_majority_loss, average_loss