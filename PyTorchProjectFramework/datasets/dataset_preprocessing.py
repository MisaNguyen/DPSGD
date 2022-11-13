from datasets import MNIST_dataset, CIFAR10_dataset

def dataset_preprocessing(dataset_name, train_kwargs, test_kwargs):
    if(dataset_name == "MNIST"):
        dataset = MNIST_dataset
        print("Processing MNIST dataset")
    elif(dataset_name == "CIFAR10"):
        dataset = CIFAR10_dataset
        print("Processing Cifar10 dataset")
    else:
        raise Exception("Invalid dataset name, try: MNIST, CIFAR10")

    train_batches, test_loader , dataset_size = dataset.shuffling_preprocessing(train_kwargs,test_kwargs)
    return train_batches, test_loader , dataset_size