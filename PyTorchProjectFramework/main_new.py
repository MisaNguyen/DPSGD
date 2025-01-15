
import torch
import torch.nn as nn
# from torchsummary import summary
import argparse
import json
import copy
import numpy as np
import os
import pickle
"""MODELS"""
# from models.densenet_model import densenet40_k12_cifar10
# from models.alexnet_model import AlexNet
# from models.alexnet_simple import AlexNet
# from models.simple_dla import SimpleDLA
from models.convnet_model import convnet
from models.nor_convnet_model import nor_convnet
from models.group_nor_convnet_model import group_nor_convnet
from models.Lenet_model import LeNet
from models.nor_Lenet_model import nor_LeNet
from models.resnet_model import ResNet18
from models.resnet_model_no_BN import ResNet18_no_BN
from models.plainnet import PlainNet18
from models.square_model import SquareNet
# from models.vgg16 import VGGNet

"""DATASETS"""

from datasets.dataset_preprocessing import dataset_preprocessing
"""UTILS"""
from utils.utils import generate_json_data_for_graph, json_to_file
from utils.utils import compute_layerwise_C, compute_layerwise_C_average_norm

"""TRAIN AND VALIDATE"""
import validate_model
import train_model_BC_FGC
import train_model_BC_FGC_v1
import train_model_BC_FGC_v2
import train_model_BC_LWGC
import train_model_BC_LWGC_v1
import train_model_BC_LWGC_v2
import train_model_BC_DLWGC
import train_model_BC_DLWGC_v1
import train_model_BC_DLWGC_v2
import train_model_BC_LWGC_classical

import train_model_IC_FGC
import train_model_IC_LWGC
import train_model_IC_DLWGC
"""OPTIMIZERS"""
import torch.optim as optim

""" OPACUS"""
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

""" SCHEDULERS"""
import torch.optim.lr_scheduler as lr_scheduler

def get_model(model_name):
    """
    Returns a model instance given a model name.

    Parameters
    ----------
    model_name : str
        The name of the model to return.

    Returns
    -------
    model : nn.Module
        An instance of the model.

    Raises
    ------
    ValueError
        If the model name is not recognized.
    """
    num_classes = 10
    if model_name == "convnet":
        model = convnet(num_classes=num_classes)
    elif model_name == "nor_convnet":
        model = nor_convnet(num_classes=num_classes)
    elif model_name == "group_nor_convnet":
        model = group_nor_convnet(num_classes=num_classes)
    elif model_name == "LeNet":
        model = LeNet()
    elif model_name == "nor_LeNet":
        model = nor_LeNet()
    elif model_name == "ResNet18":
        model = ResNet18()
    elif model_name == "ResNet18_no_BN":
        model = ResNet18_no_BN()
    elif model_name == "PlainNet18":
        model = PlainNet18()
    elif model_name == "SquareNet":
        model = SquareNet()
    return model
def load_config(file_path):
    """
    Load a configuration file.

    Parameters
    ----------
    file_path : str
        The path to the configuration file.

    Returns
    -------
    config : dict
        The configuration dictionary loaded from the file.
    """
    configs = dict()
    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)
        print(json_data.keys())
        configs = json_data
    return configs

def main():
    # Path to setting folder (Loading)
    setting_folder = "settings" 
    # settings_file = "settings_clipping_exp_cifar10_dpsgd_opacus"
    # settings_file = "settings_convnet"
    settings_file = "settings_convnet_1028"
    # settings_file = "settings_convnet_1028_lr_low"
    
    file_path = setting_folder + "/" + settings_file +".json"
    # Path to graph folder (Saving)
    out_file_path = "./graphs/new_result"    
    # CUDA
    no_cuda = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    # Dataset
    dataset_name = "CIFAR10"
    sampling_mode = "subsampling"
    shuffle_dataset = False # Shuffle dataset after each training setting
    # Model
    # model_name = "convnet"
    # model_name = "group_nor_convnet"
    model_name = "nor_convnet"
    model = get_model(model_name)
    out_file_path = out_file_path + "/" + model_name
    # Loss function
    # loss_fn = nn.CrossEntropyLoss(reduction='sum') # IMPORTANT:For v1 reduction='sum' for privacy engine
    loss_fn = nn.CrossEntropyLoss(reduction='mean') # IMPORTANT: For v2 reduction='sum' for privacy engine
    epochs = 10
    # Move model graph to GPU if enable cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    # Privacy engine
    IC = False
    BC = False
    BC_classical = True 
    #---#
    FGC = False
    LWGC = True
    DLWGC = False
    #---#
    v1 = False
    v2 = False
    if(IC and FGC):
        out_file_path = out_file_path + "/IC_FGC"
    elif(IC and LWGC):
        out_file_path = out_file_path + "/IC_LWGC"
    elif(IC and DLWGC):
        out_file_path = out_file_path + "/IC_DLWGC"
    elif(BC and FGC):
        out_file_path = out_file_path + "/BC_FGC"
        if(v1):
            out_file_path = out_file_path +  "/v1"
        elif(v2):
            out_file_path = out_file_path +  "/v2"
    elif(BC and LWGC):
        out_file_path = out_file_path + "/BC_LWGC"
        if(v1):
            out_file_path = out_file_path +  "/v1"
        elif(v2):
            out_file_path = out_file_path +  "/v2"
    elif(BC and DLWGC):
        out_file_path = out_file_path + "/BC_DLWGC"
        if(v1):
            out_file_path = out_file_path +  "/v1"
        elif(v2):
            out_file_path = out_file_path +  "/v2"
    elif (BC_classical and LWGC):
        out_file_path = out_file_path + "/BC_classical_LWGC"
    # Load configs file
    configs = load_config(file_path)
    outfile = dict()
    for config in configs.keys(): 
        optimizer = optim.SGD(model.parameters(), lr=configs[config]["learning_rate"])
        # Training/testing arguments
        if(shuffle_dataset):
            train_kwargs = {'batch_size': int(configs[config]["batch_size"]),  'shuffle': True}
        else:   
            train_kwargs = {'batch_size': int(configs[config]["batch_size"]),  'shuffle': False}
        test_kwargs = {'batch_size': int(configs[config]["test_batch_size"]), 'shuffle': False}
        # Get Dataloaders
        C_dataset_loader, train_loader, test_loader, dataset_size = dataset_preprocessing(dataset_name, train_kwargs,
                                                                     test_kwargs,sampling_mode)
        # Cuda specs
        if use_cuda:
            cuda_kwargs = {'num_workers': 2,
                        'pin_memory': True,
                        }
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)
        # StepLR =>>> new LR = old LR * gamma
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        
        print("setting:", config)
        print("setting_data:", configs[config])
        """TRAINING LOOP"""
        if(IC and FGC):
            print("Individual clipping and Full gradient Clipping")
            
            train_accuracy, test_accuracy = train_model_IC_FGC.training_loop(
                                                            n_epochs = epochs,
                                                            optimizer = optimizer,
                                                            model = model,
                                                            sigma = configs[config]["noise_multiplier"],
                                                            const_C = configs[config]["max_grad_norm"],
                                                            loss_fn = loss_fn,
                                                            scheduler=scheduler,
                                                            train_loader = train_loader,
                                                            val_loader = test_loader,
                                                            device=device
                                                        )
        elif(IC and LWGC):
            print("Individual clipping and Naive Layerwise gradient Clipping")
            
            train_accuracy, test_accuracy = train_model_IC_LWGC.training_loop(
                                                            n_epochs = epochs,
                                                            optimizer = optimizer,
                                                            model = model,
                                                            sigma = configs[config]["noise_multiplier"],
                                                            const_C = configs[config]["max_grad_norm"],
                                                            loss_fn = loss_fn,
                                                            scheduler=scheduler,
                                                            train_loader = train_loader,
                                                            val_loader = test_loader,
                                                            device=device
                                                        )
        elif(IC and DLWGC):
            print("Individual clipping and Dynamic Layerwise gradient Clipping")
            train_accuracy, test_accuracy = train_model_IC_DLWGC.training_loop(
                                                            n_epochs = epochs,
                                                            optimizer = optimizer,
                                                            model = model,
                                                            sigma = configs[config]["noise_multiplier"],
                                                            const_C = configs[config]["max_grad_norm"],
                                                            loss_fn = loss_fn,
                                                            scheduler=scheduler,
                                                            train_loader = train_loader,
                                                            val_loader = test_loader,
                                                            data_surrogate_loader = C_dataset_loader,
                                                            device=device
                                                        )
        elif(BC and FGC):
            if (v1):
                print("Batch clipping and Full gradient Clipping V1")
                train_accuracy, test_accuracy = train_model_BC_FGC_v1.training_loop(
                                                            n_epochs = epochs,
                                                            optimizer = optimizer,
                                                            model = model,
                                                            sigma = configs[config]["noise_multiplier"],
                                                            const_C = configs[config]["max_grad_norm"],
                                                            loss_fn = loss_fn,
                                                            scheduler=scheduler,
                                                            train_loader = train_loader,
                                                            val_loader = test_loader,
                                                            device=device
                                                        )
            elif(v2):
                print("Batch clipping and Full Layerwise gradient Clipping V2")
                train_accuracy, test_accuracy = train_model_BC_FGC_v2.training_loop(
                                                            n_epochs = epochs,
                                                            optimizer = optimizer,
                                                            model = model,
                                                            sigma = configs[config]["noise_multiplier"],
                                                            const_C = configs[config]["max_grad_norm"],
                                                            loss_fn = loss_fn,
                                                            scheduler=scheduler,
                                                            train_loader = train_loader,
                                                            val_loader = test_loader,
                                                            device=device
                                                        )
            else:
                print("Batch clipping and Full gradient Clipping V0")
                train_accuracy, test_accuracy = train_model_BC_FGC.training_loop(
                                                            n_epochs = epochs,
                                                            optimizer = optimizer,
                                                            model = model,
                                                            sigma = configs[config]["noise_multiplier"],
                                                            const_C = configs[config]["max_grad_norm"],
                                                            loss_fn = loss_fn,
                                                            scheduler=scheduler,
                                                            train_loader = train_loader,
                                                            val_loader = test_loader,
                                                            device=device
                                                        )
        elif(BC and LWGC):
            if (v1):
                print("Batch clipping and Naive Layerwise gradient Clipping V1")
                train_accuracy, test_accuracy = train_model_BC_LWGC_v1.training_loop(
                                                            n_epochs = epochs,
                                                            optimizer = optimizer,
                                                            model = model,
                                                            sigma = configs[config]["noise_multiplier"],
                                                            const_C = configs[config]["max_grad_norm"],
                                                            loss_fn = loss_fn,
                                                            scheduler=scheduler,
                                                            train_loader = train_loader,
                                                            val_loader = test_loader,
                                                            device=device
                                                        )
            elif(v2):
                print("Batch clipping and Naive Layerwise gradient Clipping V2")
                train_accuracy, test_accuracy = train_model_BC_LWGC_v2.training_loop(
                                                            n_epochs = epochs,
                                                            optimizer = optimizer,
                                                            model = model,
                                                            sigma = configs[config]["noise_multiplier"],
                                                            const_C = configs[config]["max_grad_norm"],
                                                            loss_fn = loss_fn,
                                                            scheduler=scheduler,
                                                            train_loader = train_loader,
                                                            val_loader = test_loader,
                                                            device=device
                                                        )
            else:
                print("Batch clipping and Naive Layerwise gradient Clipping V0")
                train_accuracy, test_accuracy = train_model_BC_LWGC.training_loop(
                                                                n_epochs = epochs,
                                                                optimizer = optimizer,
                                                                model = model,
                                                                sigma = configs[config]["noise_multiplier"],
                                                                const_C = configs[config]["max_grad_norm"],
                                                                loss_fn = loss_fn,
                                                                scheduler=scheduler,
                                                                train_loader = train_loader,
                                                                val_loader = test_loader,
                                                                device=device
                                                            )
            
        elif(BC and DLWGC):
            if (v1):
                print("Batch clipping and Dynamic Layerwise gradient Clipping V1")
                train_accuracy, test_accuracy = train_model_BC_DLWGC_v1.training_loop(
                                                            n_epochs = epochs,
                                                            optimizer = optimizer,
                                                            model = model,
                                                            sigma = configs[config]["noise_multiplier"],
                                                            const_C = configs[config]["max_grad_norm"],
                                                            loss_fn = loss_fn,
                                                            scheduler=scheduler,
                                                            train_loader = train_loader,
                                                            val_loader = test_loader,
                                                            data_surrogate_loader = C_dataset_loader,
                                                            device=device
                                                        )
            elif(v2):
                print("Batch clipping and Dynamic Layerwise gradient Clipping V2")
                train_accuracy, test_accuracy = train_model_BC_DLWGC_v2.training_loop(
                                                            n_epochs = epochs,
                                                            optimizer = optimizer,
                                                            model = model,
                                                            sigma = configs[config]["noise_multiplier"],
                                                            const_C = configs[config]["max_grad_norm"],
                                                            loss_fn = loss_fn,
                                                            scheduler=scheduler,
                                                            train_loader = train_loader,
                                                            val_loader = test_loader,
                                                            data_surrogate_loader = C_dataset_loader,
                                                            device=device
                                                        )
                
            else:
                print("Batch clipping and Dynamic Layerwise gradient Clipping V0")
                train_accuracy, test_accuracy = train_model_BC_DLWGC.training_loop(
                                                                n_epochs = epochs,
                                                                optimizer = optimizer,
                                                                model = model,
                                                                sigma = configs[config]["noise_multiplier"],
                                                                const_C = configs[config]["max_grad_norm"],
                                                                loss_fn = loss_fn,
                                                                scheduler=scheduler,
                                                                train_loader = train_loader,
                                                                val_loader = test_loader,
                                                                data_surrogate_loader = C_dataset_loader,
                                                                device=device
                                                            )
        elif(BC_classical and LWGC):
            print("Batch clipping classical SGD and Naive Layerwise gradient Clipping V0")
            configs[config]["outer_n_epochs"] = 10
            # configs[config]["outer_batch_size"] = configs[config]["batch_size"]
            configs[config]["lr_outer_initial"] = 0.5
            configs[config]["inner_n_epochs"] = 5
            configs[config]["inner_batch_size"] = 64
            # configs[config]["lr_inner_initial"] = configs[config]["learning_rate"]
            # model_tmp = get_model(model_name)
            # optimizer_tmp = torch.optim.SGD(model_tmp.parameters())
            # TESTING
            train_kwargs = {'batch_size': 10000,  'shuffle': True}
            if use_cuda:
                cuda_kwargs = {'num_workers': 2,
                            'pin_memory': True,
                            }
                train_kwargs.update(cuda_kwargs)
            C_dataset_loader, train_loader, test_loader, dataset_size = dataset_preprocessing(dataset_name, train_kwargs,
                                                                     test_kwargs,sampling_mode)
            configs[config]["max_grad_norm"] = 1000
            configs[config]["noise_multiplier"] = 0
            train_accuracy, test_accuracy = train_model_BC_LWGC_classical.training_loop(
                                                                            outer_n_epochs = configs[config]["outer_n_epochs"],
                                                                            optimizer = optimizer,
                                                                            # optimizer_tmp = optimizer_tmp,
                                                                            model = model,
                                                                            # model_tmp = model_tmp,
                                                                            loss_fn = loss_fn,
                                                                            inner_n_epochs = configs[config]["inner_n_epochs"],
                                                                            inner_batch_size = configs[config]["inner_batch_size"],
                                                                            lr_outer = configs[config]["lr_outer_initial"],
                                                                            sigma = configs[config]["noise_multiplier"],
                                                                            const_C = configs[config]["max_grad_norm"],
                                                                            train_loader = train_loader,
                                                                            val_loader = test_loader,
                                                                            device = device
                                                                        )
            
        configs[config]["train_accuracy"] = train_accuracy 
        configs[config]["test_accuracy"] = test_accuracy   
        configs[config]["model_name"] = model_name

        # break 
 
        # """
        # DECREASE C VALUE
        # """

        # if not args.ci_as_average_norm:
        #     # args.each_layer_C = compute_layerwise_C_average_norm(C_dataset_loader, dummy_model, at_epoch, device,
        #     #                                                      dummy_optimizer, args.max_grad_norm,True))
        #     if(args.enable_DP and args.enable_diminishing_gradient_norm and args.clipping == "layerwise" and not args.opacus_training):
        #         # if (args.brake_C):
        #         args.max_grad_norm = args.max_grad_norm * args.C_decay
        #         # Recompute each layer C
        #         if (args.constant_c_i):
        #             number_of_layer = get_number_of_layer(mode,model_name)
        #             args.each_layer_C = [args.max_grad_norm]*number_of_layer
        #         else:
        #             args.each_layer_C = compute_layerwise_C(C_dataset_loader, model, 1, device,
        #                                                 optimizer, args.max_grad_norm,False)
        #         print("each_layer_C", args.each_layer_C)
        # """
        # Update learning rate after each epoch
        # """
        # if (args.gamma < 1):
        #     # if(test_accuracy[-1] <= test_accuracy[-2]):
        #     args.lr = args.lr * args.gamma
        #     print("current learning rate:", args.lr)
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = args.lr
    # if (save_gradient):
    #     grad_out_file_path = out_file_path + "/grad"
    #     json_to_file(grad_out_file_path, args.load_setting, grad_array)
    # generate_json_data_for_graph(out_file_path, settings_file, train_accuracy,test_accuracy)
    isExist = os.path.exists(out_file_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(out_file_path)
        print("The new directory is created: %s" % out_file_path)
    
    print("Saving result to: ", out_file_path + "/" + settings_file + "no_noise.json")
    with open(out_file_path + "/" + settings_file + "no_noise.json", "w") as data_file:
        json.dump(configs, data_file,indent=2)
    # if args.save_model:
    #     torch.save(model.state_dict(), "cifar10_cnn.pt")

if __name__ == '__main__':
    main()