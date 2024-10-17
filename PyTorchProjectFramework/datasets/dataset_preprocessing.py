from datasets import MNIST_dataset, CIFAR10_dataset, Imagenet_dataset

def dataset_preprocessing(dataset_name, train_kwargs, test_kwargs, mode):
    """
    Preprocess the dataset, and return the loaders of the dataset

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to be preprocessed, should be one of ["MNIST", "CIFAR10", "IMAGENET"]
    train_kwargs : dict
        The keyword arguments for the training loader
    test_kwargs : dict
        The keyword arguments for the testing loader
    mode : str
        The mode of preprocessing, should be one of ["shuffling", "subsampling", "data"]
        "shuffling" mode will randomly split the dataset into batches where every sample in the dataset will appear once (Union of all batches equal the original dataset)
        "subsampling" mode will subsample the dataset to get the batches where some sample can appear more than once (the sum of size of all batches still equal to the size of original dataset)

    Returns
    -------
    C_dataset_loader : torch.utils.data.DataLoader
        The loader of the dataset for getting the layerwise clipping constants c_i
    train_loader : torch.utils.data.DataLoader
        The loader of the dataset for training the target model
    test_loader : torch.utils.data.DataLoader
        The loader of the dataset for testing the target model
    dataset_size : int
        The size of the dataset
    """
    if(dataset_name == "MNIST"):
        dataset = MNIST_dataset
        print("Processing MNIST dataset")
    elif(dataset_name == "CIFAR10"):
        dataset = CIFAR10_dataset
        print("Processing Cifar10 dataset")
    elif(dataset_name == "IMAGENET"):
        dataset = Imagenet_dataset
        print("Processing Imagenet dataset")
    else:
        raise Exception("Invalid dataset name, try: MNIST, CIFAR10")
    print("Sampling mode:", mode)
    if (mode == "shuffling"):
        # train_batches, test_loader , dataset_size = dataset.shuffling_preprocessing(train_kwargs,test_kwargs)
        C_dataset_loader, train_loader, test_loader , dataset_size = dataset.shuffling_preprocessing(train_kwargs,test_kwargs)

    elif (mode == "subsampling"):

        C_dataset_loader, train_loader, test_loader , dataset_size = dataset.subsampling_preprocessing(train_kwargs,test_kwargs)
        # return train_loader, test_loader , dataset_size
    else:
        C_dataset_loader, train_loader, test_loader , dataset_size = dataset.data_preprocessing(train_kwargs,test_kwargs)
    return C_dataset_loader, train_loader, test_loader , dataset_size
