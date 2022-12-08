from __future__ import print_function

import torch


from torchvision import datasets, transforms

import os
import sys

from datasets.data_sampler import get_data_loaders
from torch.utils.data import Dataset

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    # print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def shuffling_preprocessing(train_kwargs,test_kwargs):
    """-------------------- NORMALIZING-------------------"""

    toTensor = [
        transforms.ToTensor(),
    ]
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    train_normalize = [
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Normalize(test_mean, test_std),
    ]
    test_normalize = [
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Normalize(test_mean, test_std),
    ]
    transform_test = transforms.Compose(
        augmentations + toTensor
        + test_normalize
    )
    transform_train = transforms.Compose(
        augmentations + toTensor
        + train_normalize
    )
    testset = datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)
    # testset = MyDataset(testset,transform=transform_test)

    trainset = datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)
    print("Finished normalizing dataset.")

    # """ FULLY SHUFFLE AND DIVIDING DATASET INTO BATCHES"""
    # number_of_batches = len(trainset) // train_kwargs['batch_size']
    # batches_length = [train_kwargs['batch_size']] * number_of_batches
    # """ No drop last option"""
    # if len(trainset) % train_kwargs['batch_size'] != 0:
    #     batches_length.append(len(trainset) % train_kwargs['batch_size'])
    # """ Split train dataset into batches"""
    # train_batches = torch.utils.data.random_split(trainset, batches_length)

    train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
    print('\nTraining Set:')
    for images, labels in train_loader:
        print('Image batch dimensions:', images.size())
        print('Image label dimensions:', labels.size())
        print(labels[:10])
        break
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)
    # Checking the dataset
    print('\nTesting Set:')
    for images, labels in test_loader:
        print('Image batch dimensions:', images.size())
        print('Image label dimensions:', labels.size())
        print(labels[:10])
        break
    # return train_loader, test_loader
    dataset_size = len(trainset)
    return train_loader, test_loader, dataset_size
    # return train_batches, test_loader, dataset_size

def subsampling_preprocessing(train_kwargs,test_kwargs):
    """-------------------- NORMALIZING-------------------"""

    toTensor = [
        transforms.ToTensor(),
    ]
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    train_normalize = [
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Normalize(test_mean, test_std),
    ]
    test_normalize = [
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Normalize(test_mean, test_std),
    ]
    transform_test = transforms.Compose(
        augmentations + toTensor
        + test_normalize
    )
    transform_train = transforms.Compose(
        augmentations + toTensor
        + train_normalize
    )
    testset = datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)
    # testset = MyDataset(testset,transform=transform_test)

    trainset = datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)
    print("Finished normalizing dataset.")
    print(train_kwargs)
    del train_kwargs['shuffle']
    sampler = torch.utils.data.RandomSampler(trainset, replacement=True, num_samples=len(trainset))
    train_loader = torch.utils.data.DataLoader(trainset, sampler=sampler, **train_kwargs)
    print('\nTraining Set:')
    for images, labels in train_loader:
        print('Image batch dimensions:', images.size())
        print('Image label dimensions:', labels.size())
        print(labels[:10])
        break
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)
    # Checking the dataset
    print('\nTesting Set:')
    for images, labels in test_loader:
        print('Image batch dimensions:', images.size())
        print('Image label dimensions:', labels.size())
        print(labels[:10])
        break
    # return train_loader, test_loader
    dataset_size = len(trainset)
    return train_loader, test_loader, dataset_size



def data_preprocessing(train_kwargs,test_kwargs):
    """-------------------- NORMALIZING-------------------"""

    toTensor = [
        transforms.ToTensor(),
    ]
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    train_normalize = [
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Normalize(test_mean, test_std),
    ]
    test_normalize = [
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Normalize(test_mean, test_std),
    ]
    transform_test = transforms.Compose(
        augmentations + toTensor
        + test_normalize
    )
    transform_train = transforms.Compose(
        augmentations + toTensor
        + train_normalize
    )
    testset = datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)
    # testset = MyDataset(testset,transform=transform_test)

    trainset = datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)
    print("Finished normalizing dataset.")

    train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
    print('\nTraining Set:')
    for images, labels in train_loader:
        print('Image batch dimensions:', images.size())
        print('Image label dimensions:', labels.size())
        print(labels[:10])
        break
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)
    # Checking the dataset
    print('\nTesting Set:')
    for images, labels in test_loader:
        print('Image batch dimensions:', images.size())
        print('Image label dimensions:', labels.size())
        print(labels[:10])
        break
    # return train_loader, test_loader
    dataset_size = len(trainset)
    return train_loader, test_loader, dataset_size

if __name__ == "__main__":
    train_kwargs = {'batch_size': 16}
    test_kwargs = {'batch_size': 1000}
    # create_dataset(train_kwargs,test_kwargs)