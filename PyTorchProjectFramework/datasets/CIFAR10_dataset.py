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


def batch_clipping_preprocessing(train_kwargs,test_kwargs):
    print('==> Preparing data..')
    """
    With DP
    """
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    """
    Without DP: Normalization violates DP
    """
    toTensor = [
        transforms.ToTensor(),
    ]



    # test_mean, test_std = get_mean_and_std(testset)
    test_normalize = [
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.Normalize(test_mean, test_std),

    ]
    transform_test = transforms.Compose(
        augmentations + toTensor + test_normalize
    )
    testset = datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)


    """
    """
    transform_train = transforms.Compose(
        augmentations + toTensor + normalize
    )
    trainset = datasets.CIFAR10(
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

def individual_clipping_preprocessing(train_kwargs,test_kwargs):
    # file_dir = os.path.dirname(".")
    # print(file_dir)
    # sys.path.append(file_dir)
    print('==> Preparing data..')
    # train_transform = transforms.Compose([transforms.ToTensor()])

    # # cifar10
    # train_set = datasets.CIFAR10(root='../data/', train=True, download=True, transform=train_transform)
    # print(train_set.train_data.shape)
    # print(train_set.train_data.mean(axis=(0,1,2))/255)
    # print(train_set.train_data.std(axis=(0,1,2))/255)
    # # (50000, 32, 32, 3)
    # # [0.49139968  0.48215841  0.44653091]
    # # [0.24703223  0.24348513  0.26158784]

    # # cifar100
    # train_set = datasets.CIFAR100(root='../data/', train=True, download=True, transform=train_transform)
    # print(train_set.train_data.shape)
    # print(train_set.train_data.mean(axis=(0,1,2))/255)
    # print(train_set.train_data.std(axis=(0,1,2))/255)
    # # (50000, 32, 32, 3)
    # # [0.50707516  0.48654887  0.44091784]
    # # [0.26733429  0.25643846  0.27615047]
    #
    # # mnist
    # train_set = datasets.MNIST(root='../data/', train=True, download=True, transform=train_transform)
    # print(list(train_set.train_data.size()))
    # print(train_set.train_data.float().mean()/255)
    # print(train_set.train_data.float().std()/255)
    # # [60000, 28, 28]
    # # 0.1306604762738429
    # # 0.30810780717887876

    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    """
    With DP
    """
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
    ]
    """
    Without DP: Normalization violates DP
    """
    toTensor = [
        transforms.ToTensor(),
    ]
    transform_train = transforms.Compose(
        augmentations + toTensor + normalize
    )


    # test_mean, test_std = get_mean_and_std(testset)
    test_normalize = [
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.Normalize(test_mean, test_std),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
    ]
    transform_test = transforms.Compose(
        augmentations
    )
    testset = datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)
    """
    """
    trainset = datasets.CIFAR10(
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

    test_normalize = [
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.Normalize(test_mean, test_std),
    ]
    transform_test = transforms.Compose(
        augmentations + toTensor + test_normalize
    )
    testset = datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)

    train_normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
    ]
    transform_train = transforms.Compose(
        augmentations + train_normalize
    )
    trainset = datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)
    # testset = datasets.CIFAR10(
    #     root='../data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)

    # Checking the dataset
    # print('Training Set:\n')
    # for images, labels in train_loader:
    #     print('Image batch dimensions:', images.size())
    #     print('Image label dimensions:', labels.size())
    #     print(labels[:10])
    #     break
    #
    # print('\nTesting Set:')
    # for images, labels in test_loader:
    #     print('Image batch dimensions:', images.size())
    #     print('Image label dimensions:', labels.size())
    #     print(labels[:10])
    #     break
    # return train_loader, test_loader
    dataset_size = len(trainset)
    return train_loader, test_loader, dataset_size

if __name__ == "__main__":
    train_kwargs = {'batch_size': 16}
    test_kwargs = {'batch_size': 1000}
    # create_dataset(train_kwargs,test_kwargs)