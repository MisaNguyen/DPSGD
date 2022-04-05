from __future__ import print_function

import torch


from torchvision import datasets, transforms

# functions to show an image



def create_dataset(train_kwargs,test_kwargs):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # transform_train = transforms.Compose([transforms.Resize((70, 70)),
    #                                        transforms.RandomCrop((64, 64)),
    #                                        transforms.ToTensor(),
    #                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #                                       ])
    #
    # transform_test = transforms.Compose([transforms.Resize((70, 70)),
    #                                   transforms.CenterCrop((64, 64)),
    #                                   transforms.ToTensor(),
    #                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #                                      ])

    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # transform_train = transform
    # transform_test = transform
    trainset = datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)

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
    print('\nTesting Set:')
    for images, labels in train_loader:
        print('Image batch dimensions:', images.size())
        print('Image label dimensions:', labels.size())
        print(labels[:10])
        break
    return train_loader, test_loader
