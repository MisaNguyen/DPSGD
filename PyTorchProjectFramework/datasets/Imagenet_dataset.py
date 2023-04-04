import torch


from torchvision import datasets, transforms

import os
import sys

from datasets.data_sampler import get_data_loaders
from torch.utils.data import Dataset

def shuffling_preprocessing(train_kwargs,test_kwargs):
    """-------------------- NORMALIZING-------------------"""
    data_path = "F:\Download\imagenet"
    toTensor = [
        transforms.ToTensor(),
    ]
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    train_normalize = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #transforms.Normalize(test_mean, test_std),
    ]
    test_normalize = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    # trainset = datasets.ImageNet(
    #     root=data_path,split='train',   transform=transform_train)
    # testset = datasets.ImageNet(
    #     root=data_path,split='val',   transform=transform_test)
    trainset = datasets.ImageFolder(
        root=traindir,   transform=transform_train)
    testset = datasets.ImageFolder(
        root=valdir,   transform=transform_test)
    # testset = MyDataset(testset,transform=transform_test)


    print("Finished normalizing dataset.")


    # """ FULLY SHUFFLE AND DIVIDING DATASET INTO BATCHES"""
    # number_of_batches = len(trainset) // train_kwargs['batch_size']
    # batches_length = [train_kwargs['batch_size']] * number_of_batches
    # """ No drop last option"""
    # if len(trainset) % train_kwargs['batch_size'] != 0:
    #     batches_length.append(len(trainset) % train_kwargs['batch_size'])
    # """ Split train dataset into batches"""
    # train_batches = torch.utils.data.random_split(trainset, batches_length)
    training_len = 9*len(trainset)//10

    trainset , C_dataset = torch.utils.data.random_split(trainset, [training_len,len(trainset)-training_len])
    train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
    C_dataset_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
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
    return C_dataset_loader, train_loader, test_loader, dataset_size
    # return train_batches, test_loader, dataset_size

def subsampling_preprocessing(train_kwargs,test_kwargs):
    """-------------------- NORMALIZING-------------------"""
    data_path = "F:\Download\imagenet"
    toTensor = [
        transforms.ToTensor(),
    ]
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    train_normalize = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #transforms.Normalize(test_mean, test_std),
    ]
    test_normalize = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    trainset = datasets.ImageFolder(
        root=traindir,   transform=transform_train)
    testset = datasets.ImageFolder(
        root=valdir,   transform=transform_test)
    # testset = MyDataset(testset,transform=transform_test)


    print("Finished normalizing dataset.")
    training_len = 9*len(trainset)//10
    trainset , C_dataset = torch.utils.data.random_split(trainset, [training_len,len(trainset)-training_len])

    del train_kwargs['shuffle']
    sampler = torch.utils.data.RandomSampler(trainset, replacement=True, num_samples=len(trainset))
    train_loader = torch.utils.data.DataLoader(trainset, sampler=sampler, **train_kwargs)
    C_dataset_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
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
    return C_dataset_loader, train_loader, test_loader, dataset_size



def data_preprocessing(train_kwargs,test_kwargs):
    """-------------------- NORMALIZING-------------------"""
    data_path = "F:\Download\imagenet"
    toTensor = [
        transforms.ToTensor(),
    ]
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    train_normalize = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #transforms.Normalize(test_mean, test_std),
    ]
    test_normalize = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    # testset = MyDataset(testset,transform=transform_test)

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    trainset = datasets.ImageFolder(
        root=traindir,   transform=transform_train)
    testset = datasets.ImageFolder(
        root=valdir,   transform=transform_test)
    print("Finished normalizing dataset.")
    training_len = 9*len(trainset)//10
    trainset , C_dataset = torch.utils.data.random_split(trainset, [training_len,len(trainset)-training_len])
    train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
    C_dataset_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
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
    return C_dataset_loader, train_loader, test_loader, dataset_size

if __name__ == "__main__":
    train_kwargs = {'batch_size': 16}
    test_kwargs = {'batch_size': 1000}
    # create_dataset(train_kwargs,test_kwargs)