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
    mean = torch.zeros(1)
    std = torch.zeros(1)
    # print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        # input(inputs.shape)
        mean += inputs[:,0,:,:].mean()
        std += inputs[:,0,:,:].std()
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


def batch_clipping_preprocessing(train_kwargs,test_kwargs):
    """
    With DP
    """
    toTensor = [
        transforms.ToTensor(),
    ]
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    testset = datasets.MNIST(
        root='../data', train=False, download=True, transform=transforms.Compose(toTensor))
    test_mean, test_std = get_mean_and_std(testset)
    test_normalize = [
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(test_mean, test_std),
    ]
    transform_test = transforms.Compose(
        augmentations + test_normalize
    )
    testset = MyDataset(testset,transform=transform_test)

    trainset = datasets.MNIST(
        root='../data', train=True, download=True, transform=transforms.Compose(toTensor))
    number_of_batches = len(trainset) // train_kwargs['batch_size']
    batches_length = [train_kwargs['batch_size']] * number_of_batches
    """ No drop last option"""
    if len(trainset) % train_kwargs['batch_size'] != 0:
        batches_length.append(len(trainset) % train_kwargs['batch_size'])
    """ Split train dataset into batches"""
    batches = torch.utils.data.random_split(trainset, batches_length)

    # mean, std = get_mean_and_std(trainset)
    # print(mean, std)
    # input()
    """ Normalize each batch"""
    print("normalizing batches...")
    for idx, batch in enumerate(batches):
        # mean, std = get_mean_and_std(batch)
        print("batch_ID: %f" % (idx))
        # print("mean:", mean)
        # print("std:", std)

        # print(mean, std)
        # input()
        train_normalize = [
            # transforms.Normalize(mean, std),
            # transforms.Normalize((0.1307,), (0.3081,))
            transforms.Normalize((0.15,), (0.3,))
        ]
        transform_train = transforms.Compose(
            augmentations + train_normalize
        )
        batches[idx] = MyDataset(batches[idx],transform=transform_train)
        # train_loader = torch.utils.data.DataLoader(batches[idx], batch_size=1, shuffle=True) # Load each data
        # for sample_idx, (data,target) in enumerate(train_loader):
        #     pass
    print("Finished normalizing batches.")
    # MyDataset
    # train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)
    # print(train_kwargs['batch_size'])
    # print(train_kwargs['microbatch_size'])
    # minibatch_loader, microbatch_loader = get_data_loaders(train_kwargs['batch_size'],
    #                                                        train_kwargs['microbatch_size'],
    #                                                        len(trainset)) # iteration = training set size = 1 epoch
    # train_minibatch_loader = minibatch_loader(trainset)

    # Checking the dataset
    # print('Training Set:\n')
    # for images, labels in train_loader:
    #     print('Image batch dimensions:', images.size())
    #     print('Image label dimensions:', labels.size())
    #     print(labels[:10])
    #     break


    # # Checking the dataset
    # print('\nValidation Set:')
    # for images, labels in valid_loader:
    #     print('Image batch dimensions:', images.size())
    #     print('Image label dimensions:', labels.size())
    #     print(labels[:10])
    #     break

    # Checking the dataset
    print('\nTesting Set:')
    for images, labels in test_loader:
        print('Image batch dimensions:', images.size())
        print('Image label dimensions:', labels.size())
        print(labels[:10])
        break
    # return train_loader, test_loader
    dataset_size = len(trainset)
    return batches, test_loader, dataset_size

def individual_clipping_preprocessing(train_kwargs,test_kwargs):
    print('==> Preparing data..')
    """
    With DP
    """
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        transforms.Normalize((0.15,), (0.3,))
    ]
    """
    Without DP: Normalization violates DP
    """
    toTensor = [
        transforms.ToTensor(),
    ]
    transform_train = transforms.Compose(
        augmentations + toTensor
    )

    testset = datasets.MNIST(
        root='../data', train=False, download=True, transform=transforms.Compose(toTensor))
    test_mean, test_std = get_mean_and_std(testset)
    test_normalize = [
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(test_mean, test_std),
    ]
    transform_test = transforms.Compose(
        augmentations
    )
    testset = MyDataset(testset,transform=transform_test)


    """
    """
    trainset = datasets.MNIST(
        root='../data', train=True, download=True, transform=transform_train)


    train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)
    # print(train_kwargs['batch_size'])
    # print(train_kwargs['microbatch_size'])
    # minibatch_loader, microbatch_loader = get_data_loaders(train_kwargs['batch_size'],
    #                                                        train_kwargs['microbatch_size'],
    #                                                        len(trainset)) # iteration = training set size = 1 epoch
    # train_minibatch_loader = minibatch_loader(trainset)

    # Checking the dataset
    print('Training Set:\n')
    for images, labels in train_loader:
        print('Image batch dimensions:', images.size())
        print('Image label dimensions:', labels.size())
        print(labels[:10])
        break


    # # Checking the dataset
    # print('\nValidation Set:')
    # for images, labels in valid_loader:
    #     print('Image batch dimensions:', images.size())
    #     print('Image label dimensions:', labels.size())
    #     print(labels[:10])
    #     break

    # Checking the dataset
    # print('\nTesting Set:')
    # for images, labels in test_loader:
    #     print('Image batch dimensions:', images.size())
    #     print('Image label dimensions:', labels.size())
    #     print(labels[:10])
    #     break
    # return train_loader, test_loader
    dataset_size = len(trainset)
    return train_loader, test_loader, dataset_size

def minibatch_SGD_preprocessing(train_kwargs,test_kwargs):
    toTensor = [
        transforms.ToTensor(),
    ]
    augmentations = [
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    testset = datasets.MNIST(
        root='../data', train=False, download=True, transform=transforms.Compose(toTensor))
    test_mean, test_std = get_mean_and_std(testset)
    test_normalize = [
        # transforms.Normalize((0.1307,), (0.3081,))
        transforms.Normalize(test_mean, test_std),
    ]
    transform_test = transforms.Compose(
        augmentations + test_normalize
    )
    testset = MyDataset(testset,transform=transform_test)

    train_normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.15,), (0.3,))
        # transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
    ]
    transform_train = transforms.Compose(
        augmentations + train_normalize
    )
    trainset = datasets.MNIST(
        root='../data', train=True, download=True, transform=transform_train)
    # testset = datasets.MNIST(
    #     root='../data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)


    dataset_size = len(trainset)
    return train_loader, test_loader, dataset_size

if __name__ == "__main__":
    train_kwargs = {'batch_size': 16}
    test_kwargs = {'batch_size': 1000}
    # create_dataset(train_kwargs,test_kwargs)