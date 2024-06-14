import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
import torchvision.datasets as datasets
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from collections import Counter
import numpy as np

__all__ = ['load_train_test_dataloader']
def load_train_test_dataloader(train_config, device):
    transform_func = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
    ])
    if train_config['dataset'] == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='../cifar10_dataset', train=True, download=True, transform=transform_func)
        test_dataset = datasets.CIFAR10(root='../cifar10_dataset', train=False, download=True, transform=transform_func)
    elif train_config['dataset'] == 'MNIST':
        train_dataset = datasets.CIFAR10(root='../data/mnist_dataset', train=True, download=True, transform=transform_func)
        test_dataset = datasets.CIFAR10(root='../data/mnist_dataset', train=False, download=True, transform=transform_func)
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