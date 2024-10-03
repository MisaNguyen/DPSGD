
import torch
# from torchsummary import summary
import argparse
import json
import copy
import numpy as np
import os
"""MODELS"""
# from models.densenet_model import densenet40_k12_cifar10
from models.alexnet_model import AlexNet
# from models.alexnet_simple import AlexNet
# from models.simple_dla import SimpleDLA
from models.convnet_model import convnet
from models.nor_convnet_model import nor_convnet
from models.group_nor_convnet_model import group_nor_convnet
from models.GN_BN_convnet_model import GN_BN_convnet
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
import train_model

"""OPTIMIZERS"""
import torch.optim as optim

""" OPACUS"""
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

def refresh_model(args,
                  model_name, device,
                  opt_name,
                  lr, pre_train=True):
    # Reset parameters
    args.lr = lr
    model = get_model(model_name,device)
    optimizer = get_optimizer(opt_name,model,lr)
    return model, optimizer

def get_model(model_name, device):
    if model_name == "AlexNet":
        return AlexNet(num_classes=10).to(device)
    elif model_name == "nor_convnet":
        return nor_convnet(num_classes=10).to(device)
    elif model_name == "convnet":
        return convnet(num_classes=10).to(device)
    elif model_name == "group_nor_convnet":
        return group_nor_convnet(num_classes=10).to(device)
    elif model_name == "GN_BN_convnet":
        return GN_BN_convnet(num_classes=10).to(device)
    elif model_name == "ResNet18":
        return ResNet18(num_classes=10).to(device)
    elif model_name == "ResNet18_no_BN":
        return ResNet18_no_BN(num_classes=10).to(device)
    elif model_name == "LeNet":
        return LeNet().to(device)
    elif model_name == "nor_LeNet":
        return nor_LeNet().to(device)
    else:
        raise Exception("Unknown model name")
    return None

def get_optimizer(opt_name,model,lr):
    if opt_name == "SGD":

        # optimizer = optim.SGD(
        #     model.parameters(),
        #     lr=args.lr,
        #     # momentum=args.momentum,
        #     # weight_decay=args.weight_decay,
        # )
        optimizer= optim.SGD(params=model.parameters(),
                             # [
                             #     {"params": model.layer1.parameters(), "lr": args.lr},
                             #     {"params": model.layer2.parameters(),"lr": args.lr},
                             #     {"params": model.layer3.parameters(), "lr": args.lr},
                             #     {"params": model.layer4.parameters(), "lr": args.lr},
                             # ],
                             lr=lr,
                             )
    elif opt_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        # print(args.optimizer)
        raise NotImplementedError("Optimizer not recognized. Please check spelling")
    return optimizer
def get_number_of_layer(model, model_name):
    number_of_layer = load_number_of_layer(model_name)
    if (number_of_layer == None):
        number_of_layer = 0
        for layer_idx, (name, param) in enumerate(model.named_parameters()):
            if param.requires_grad:
                number_of_layer = number_of_layer + 1
        save_number_of_layer(model_name, number_of_layer)
    return number_of_layer

def load_number_of_layer(model_name):
    isExist = os.path.exists("num_layers.json")
    if (isExist):
        with open("num_layers.json","r") as json_file:
            json_data = json.load(json_file)
            if model_name in json_data:
                number_of_layer = json_data[model_name]
            else:
                return None
    else:
        return None
    return number_of_layer


def save_number_of_layer(model_name, number_of_layer):
    f = open("num_layers.json")
    json_data = json.load(f)
    f.close()
    with open("num_layers.json","w") as json_file:
        json_data[model_name] = number_of_layer
        json.dump(json_data, json_file,indent=2)
    print(json_data)


def add_noise_to_model(args, model, device, train_loader, optimizer):
    train_accuracy = 0
    return train_accuracy

def main():
    parser = argparse.ArgumentParser(description='DPSGD framework parser')
    args = parser.parse_args()
    # Training settings
    """
    Define sampling mode here
    """
    mode = "subsampling"
    # mode = "shuffling"
    """
    Define clipping mode here
    """
    enable_individual_clipping = True
    enable_batch_clipping = False
    enable_classical_BC = False
    """
    Define stepsize mode here
    """
    train_with_constant_step_size = False
    """
    Toggle on/off noise multiplier (sigma) discount for full gradient clipping
    """
    sigma_discount_on = False
    # mode = None
    setting_folder = "settings"
    # settings_file = "settings_lost_func_grid_search_sigma2_1"
    # settings_file = "settings_best_settings_lost_func_grid_search_1"
    settings_file = "settings_clipping_exp_cifar10_dpsgd_opacus"
    """ Random seed"""
    seed = 10
    torch.manual_seed(seed)

    """Dataset args"""
    shuffle_dataset = True
    batch_size = 64
    test_batch_size = 1000
    if(shuffle_dataset):
        train_kwargs = {'batch_size': batch_size,  'shuffle': True}
    else:
        train_kwargs = {'batch_size': batch_size,  'shuffle': False}
    test_kwargs = {'batch_size': test_batch_size, 'shuffle': False}
    """ Check gpu availablity """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        cuda_kwargs = {'num_workers': 0, # TODO: default is 2, 0 for fixing opacus multiprocessing
                       'pin_memory': True,
                       }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    """Model"""
    # model_name = "AlexNet"
    # model_name = "nor_convnet"
    # model_name = "convnet"
    # model_name = "GN_BN_convnet"
    # model_name = "resnet18"
    model_name = "group_nor_convnet"
    # model_name = "resnet18_no_BN"
    # model_name = "plainnet18"
    # model_name = "LeNet"
    # model_name = "nor_LeNet"
    # model_name = "squarenet"
    model = get_model(model_name,device)
    """ Learning Rate """
    args.lr = 0.1
    args.gamma = 0.9
    """ Optimizer """
    args.optimizer = "SGD"
    optimizer = get_optimizer(args.optimizer,model,args.lr)
    """ Print specs"""
    print("Training with model:", model_name)
    print("Opimizer:", optimizer)
    print("Learning rate", args.lr)
    print("batch size", batch_size)
    print("test batch size", test_batch_size)


    # print('Initializing visualization...')
    # visualizer = Visualizer({"name": "MNIST DPSGD"})
    visualizer = None
    train_accuracy = []
    test_accuracy = []
    epochs = 50

    args.save_gradient = False
    args.dry_run = False # Run for 1 iteration
    args.log_interval = 1000
    model_reloaded = False
    """Dataset preprocessing"""
    dataset_name = "MNIST"
    dataset_name = "CIFAR10"
    C_dataset_loader, train_loader, test_loader, dataset_size = dataset_preprocessing(dataset_name, train_kwargs,
                                                                                      test_kwargs,mode)
    print("dataset_size=", dataset_size)
    pre_train = False
    if (pre_train == True):
        target_acc = 30
        """Training base model"""
        # model_save_path = dataset_name +"_"+ model_name + "_" +"base_model.pt"
        model_save_path = dataset_name +"_"+ model_name + "_" +"base_model_" + str(target_acc) +".pt"
        if(os.path.exists(model_save_path)):
            print("Loading base model at:", model_save_path)
            model.load_state_dict(torch.load(model_save_path))
        else:
            print("Training base model")
            model_reloaded = True
            # Train baseline model
            for epoch in range(1, epochs + 1):
                print("epoch", epoch)
                train_acc, gradient_stats = train_model.train(args, model, device, train_loader, optimizer,epoch)
                train_accuracy.append(train_acc)
                test_accuracy.append(validate_model.test(model, device, test_loader))
                if (test_accuracy[-1] > target_acc/100): # break if model acc > target_acc/100
                    break

                """
                Update learning rate after each epoch
                """
                if (args.gamma < 1):
                    # if(test_accuracy[-1] <= test_accuracy[-2]):
                    args.lr = args.lr * args.gamma
                    print("current learning rate:", args.lr)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = args.lr
            torch.save(model.state_dict(), model_save_path)
    """
    Train base model with noise (IC/BC)
    """
    private_epochs = 30
    start_lr = 0.001
    args.noise_multiplier = 0.05
    args.max_grad_norm = 1.2
    args.loss_multi = 1
    # args.clipping = "layerwise"
    args.clipping = "all"
    args.constant_c_i = False
    args.ci_as_average_norm = False
    C_arr = [0.001 * pow(2,i) for i in range(1,17)]
    # C_arr = [0.001 * pow(2,i) for i in range(16,17)]
    base_model_test_acc = validate_model.test(model, device, test_loader)
    # print("current model test acc", base_model_test_acc)
    graph_data = [{"base_model_test_acc": base_model_test_acc,
                   "sigma": args.noise_multiplier,
                   "private_epochs": private_epochs}]
    args.opacus_training = False

    for C in C_arr:
        if (model_reloaded == False):
            del model
            del optimizer
            model,optimizer = refresh_model(args,
                                            model_name, device,
                                            args.optimizer,
                                            start_lr, pre_train=False)
            if(pre_train != False):
                """ Reload base model"""
                print("HERE")
                print("C = ", C)
                print("Loading base model at:", model_save_path)
                model.load_state_dict(torch.load(model_save_path))
                # args.lr = start_lr
                model_reloaded = True
            else:
                # Clear model and optimizer values
                # del model
                # del optimizer
                # model,optimizer = refresh_model(args,
                #                                 model_name, device,
                #                                 args.optimizer,
                #                                 start_lr, pre_train=False)
                model_reloaded = True
        args.max_grad_norm = C
        train_accuracy_IC = []
        test_accuracy_IC = []
        train_accuracy_BC = []
        test_accuracy_BC = []


        """ Get lay erwise c_i"""
        if(args.opacus_training):
            privacy_engine = PrivacyEngine()
            # clipping = "per_layer" if args.clipping=="layerwise" else "flat"
            if (args.clipping=="layerwise"):
                clipping = "per_layer"
                n_layers = len(
                    [(n, p) for n, p in model.named_parameters() if p.requires_grad]
                )
                max_grad_norm = [
                                    args.max_grad_norm / np.sqrt(n_layers)
                                ] * n_layers
            else:
                clipping = "flat"
                max_grad_norm = args.max_grad_norm
            model.train() # Enable train mode
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.noise_multiplier,
                max_grad_norm=max_grad_norm,
                clipping=clipping,
            )
        elif(args.clipping == "layerwise" or args.clipping == "weight_FGC"):
            if (args.constant_c_i):
                number_of_layer = get_number_of_layer(model,model_name)
                args.each_layer_C = [args.max_grad_norm]*number_of_layer
            else:
                at_epoch = 1
                dummy_model = copy.deepcopy(model)
                # print("HERE")
                # print(validate_model.test(dummy_model, device, test_loader))
                # print(args.lr)
                dummy_optimizer = get_optimizer(args.optimizer,dummy_model ,start_lr)
                if(args.ci_as_average_norm):
                    args.each_layer_C = compute_layerwise_C_average_norm(C_dataset_loader, dummy_model, at_epoch, device,
                                                                         dummy_optimizer, args.max_grad_norm,False)
                else:
                    args.each_layer_C = compute_layerwise_C(C_dataset_loader, dummy_model, at_epoch, device,
                                                            dummy_optimizer, args.max_grad_norm,False)
            print("each layer C", args.each_layer_C)
        """ IC training"""
        print("IC TRAINING")
        args.microbatch_size = 1 # IC
        if (model_reloaded == False):
            print("Cleaning model and optimizer")
            del model
            del optimizer
            model,optimizer = refresh_model(args,
                                            model_name, device,
                                            args.optimizer,
                                            start_lr, pre_train=False)
            if(pre_train != False):
                """ Reload base model"""
                print("Loading base model at:", model_save_path)
                model.load_state_dict(torch.load(model_save_path))
                # args.lr = start_lr
                # model_reloaded = True
            # else:
                # print("Cleaning model and optimizer")
                # del model
                # del optimizer
                # model,optimizer = refresh_model(args,
                #                                 model_name, device,
                #                                 args.optimizer,
                #                                 start_lr, pre_train=False)
            model_reloaded = True
        """ """
        for epoch in range(1, private_epochs +1):
            if(args.opacus_training):
                train_acc, _ = train_model.train(args, model, device, train_loader, optimizer,epoch)
                train_accuracy_IC.append(train_acc)
            else:
                train_accuracy_IC.append(train_model.DP_train(args, model, device, train_loader, optimizer))
            test_accuracy_IC.append(validate_model.test(model, device, test_loader))
            """
            Update learning rate after each epoch
            """

            if (args.gamma < 1):
                # if(test_accuracy[-1] <= test_accuracy[-2]):
                args.lr = args.lr * args.gamma
                print("current learning rate:", args.lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = args.lr
        model_reloaded = False # RELOAD MODEL AFTER TRAINING IC
        # """ BC training"""
        # args.microbatch_size = batch_size # BC
        # """ Reload base model"""
        # print("BC TRAINING")
        # if (model_reloaded == False):
        #     del model
        #     del optimizer
        #     model,optimizer = refresh_model(args,
        #                                     model_name, device,
        #                                     args.optimizer,
        #                                     start_lr, pre_train=False)
        #     if(pre_train != False):
        #         """ Reload base model"""
        #         print("Loading base model at:", model_save_path)
        #         model.load_state_dict(torch.load(model_save_path))
        #         # args.lr = start_lr
        #         model_reloaded = True
        #     else:
        #         print("Cleaning model and optimizer")
        #         # del model
        #         # del optimizer
        #         # model,optimizer = refresh_model(args,
        #         #                     model_name, device,
        #         #                     args.optimizer,
        #         #                     start_lr, pre_train=False)
        #         model_reloaded = True
        # """ """
        # for epoch in range(1, private_epochs +1):
        #     train_accuracy_BC.append(train_model.DP_train(args, model, device, train_loader, optimizer))
        #     test_accuracy_BC.append(validate_model.test(model, device, test_loader))
        #     """
        #     Update learning rate after each epoch
        #     """
        #     if (args.gamma < 1):
        #         # if(test_accuracy[-1] <= test_accuracy[-2]):
        #         args.lr = args.lr * args.gamma
        #         print("current learning rate:", args.lr)
        #         for param_group in optimizer.param_groups:
        #             param_group["lr"] = args.lr
        # # Reload model
        # model_reloaded = False
        """----------------------------------------------------"""
        graph_data.append({
            "C": C,
            "IC_train": train_accuracy_IC,
            "IC_test": test_accuracy_IC,
            "BC_train": train_accuracy_BC,
            "BC_test": test_accuracy_BC,
        })
        """--------------------------------------------------------------------"""
        """ Record data"""
        if(args.opacus_training):
            out_file_path = "./graphs/utility_graph/opacus/" + model_name
        else:
            out_file_path = "./graphs/utility_graph/" + model_name

        isExist = os.path.exists(out_file_path)

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(out_file_path)
            print("The new directory is created: %s" % out_file_path)

        if (pre_train):
            out_file_path=out_file_path + '/utility_graph_data_' + dataset_name + "_" + args.clipping
            out_file_path=out_file_path+ "_" + str(target_acc)
            if(args.noise_multiplier == 0):
                out_file_path=out_file_path+ "_noNoise"
            out_file_path=out_file_path + '.json'
        else:
            # out_file_path=out_file_path + '/utility_graph_data_' + dataset_name + "_" + args.clipping + "_no_pre_train" +'.json'
            out_file_path=out_file_path + '/utility_graph_data_' + dataset_name + "_" + args.clipping
            out_file_path=out_file_path+  '_no_pre_train'
            if(args.noise_multiplier == 0):
                out_file_path=out_file_path+ "_noNoise"
            out_file_path=out_file_path + '.json'
        with open(out_file_path , "w") as data_file:
            json.dump(graph_data, data_file,indent=2)
    # generate_json_data_for_graph(out_file_path, "IC", train_accuracy_IC,test_accuracy_IC)
    # generate_json_data_for_graph(out_file_path, "BC", train_accuracy_BC,test_accuracy_BC)
    # if(args.enable_DP == True):
    #     out_file_path = "./graphs/data_neurips/" + settings_file +  "/" + model_name + "/" + args.optimizer + "/" + str(args.clipping)
    #     if (args.clipping == "layerwise" and args.constant_c_i == True):
    #         out_file_path = out_file_path + "/constant_c_i"
    # else:
    #     out_file_path = "./graphs/data_neurips/" + settings_file +  "/" + model_name + "/" + args.optimizer
    # Get training and teopasting data loaders




if __name__ == '__main__':
    main()