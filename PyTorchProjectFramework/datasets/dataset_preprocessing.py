from datasets import MNIST_dataset, CIFAR10_dataset

def dataset_preprocessing(dataset_name, train_kwargs, test_kwargs, enable_DP,enable_diminishing_gradient_norm,enable_individual_clipping):
    if(dataset_name == "MNIST"):
        dataset = MNIST_dataset
    elif(dataset_name == "CIFAR10"):
        dataset = CIFAR10_dataset
    else:
        raise Exception("Invalid dataset name, try: MNIST, CIFAR10")
    if enable_DP:
        # privacy_engine = None
        if (enable_diminishing_gradient_norm == True):
            pass
        if (enable_individual_clipping == True):
            train_loader, test_loader, dataset_size = dataset.individual_clipping_preprocessing(train_kwargs,test_kwargs)
        else:
            train_loader, test_loader, dataset_size = dataset.batch_clipping_preprocessing(train_kwargs,test_kwargs)

    else:
        train_loader, test_loader, dataset_size = dataset.minibatch_SGD_preprocessing(train_kwargs,test_kwargs)

    return train_loader, test_loader, dataset_size