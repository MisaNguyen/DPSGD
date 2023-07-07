
import torch
# from torchsummary import summary
import argparse
import json
import copy
import numpy as np
import os
"""MODELS"""
# from models.densenet_model import densenet40_k12_cifar10
# from models.alexnet_model import AlexNet
# from models.alexnet_simple import AlexNet
# from models.simple_dla import SimpleDLA
from models.convnet_model import convnet
from models.nor_convnet_model import nor_convnet
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

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--microbatch-size', type=int, default=1, metavar='MS',
                        help='input microbatch batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--iterations', type=int, default=1000, metavar='N',
                        help='number of iterations to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--noise-multiplier', type=float, default=1.0, metavar='NM',
                        help='Noise multiplier (default: 1.0)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, metavar='MGN',
                        help='Max gradian norm (default: 1.0)')
    parser.add_argument('--optimizer', type=str, default="SGD", metavar='O',
                        help='Name of optimizer (example: SGD, DPSGD,...)')
    parser.add_argument('--load-setting', type=str, default="", metavar='LS',
                        help='Name of setting (example: setting_1, setting_2,...')
    parser.add_argument('--enable-diminishing-gradient-norm', type=bool, default=False, metavar='DGN',
                        help='Enable diminishing gradient norm mode')
    parser.add_argument('--enable-individual-clipping', type=bool, default=False, metavar='IC',
                        help='Enable individual clipping mode')
    parser.add_argument('--enable-batch-clipping', type=bool, default=False, metavar='IC',
                        help='Enable batch clipping mode')
    args = parser.parse_args()
    #Add setting path here
    """
    Define sampling mode here
    """
    mode = "subsampling"
    # mode = "shuffling"
    """
    Define clipping mode here
    """
    enable_individual_clipping = False
    enable_batch_clipping = True
    enable_classical_BC = False
    """
    Define stepsize mode here
    """
    train_with_constant_step_size = True
    """
    Toggle on/off noise multiplier (sigma) discount for full gradient clipping
    """
    sigma_discount_on = False
    # mode = None
    settings_file = "settings_best_settings"
    logging = True

    if (mode != None):
        settings_file = settings_file + "_" + mode
    # if(args.enable_DP):
    if(enable_individual_clipping):
        settings_file = settings_file + "_IC"
    elif(enable_batch_clipping):
        settings_file = settings_file + "_BC"
    elif(enable_classical_BC):
        settings_file = settings_file + "_classical"
    if(train_with_constant_step_size):
        settings_file = settings_file + "_css"
    print("Running setting: %s.json" % settings_file)
    if(args.load_setting != ""):
        with open(settings_file +".json", "r") as json_file:
            json_data = json.load(json_file)
            setting_data = json_data[args.load_setting]
            # Loading data
            args.batch_size = int(setting_data["batch_size"])
            args.microbatch_size = int(setting_data["microbatch_size"])
            args.test_batch_size = setting_data["test_batch_size"]
            args.iterations = setting_data["iterations"]
            args.lr = setting_data["learning_rate"]
            args.gamma = setting_data["gamma"]
            args.no_cuda = setting_data["no_cuda"]
            args.dry_run = setting_data["dry_run"]
            args.seed = setting_data["seed"]
            args.log_interval = setting_data["log_interval"]
            args.save_model = setting_data["save_model"]
            args.noise_multiplier = setting_data["noise_multiplier"]
            print("sigma=",args.noise_multiplier)
            args.max_grad_norm = setting_data["max_grad_norm"]
            args.optimizer = setting_data["optimizer"]
            args.enable_diminishing_gradient_norm = False
            # args.enable_individual_clipping = setting_data["is_individual_clipping"]
            # args.enable_batch_clipping = False
            # args.enable_DP = setting_data["enable_DP"]
            args.enable_DP = True #TODO: Change here before upload to github
            # args.clip_per_layer = False #TODO: add to setting file
            # args.secure_rng = False #TODO: add to setting file
            args.shuffle_dataset = True
            # args.is_partition_train = False
            args.mode = setting_data["data_sampling"]
            # args.clipping = "layerwise"#TODO: add to setting file
            # args.clipping = "all"
            args.clipping = "weight_FGC"
            args.C_decay = 0.9
            # args.dataset_name = "MNIST"
            args.dataset_name = "CIFAR10"#TODO: add to setting file
            # args.dataset_name = "Imagenet"#TODO: add to setting file
            args.opacus_training = False
            args.save_gradient = False
            args.constant_c_i = False
            # args.classicalSGD = False
            args.ci_as_average_norm = False
            # args.brake_C = True
    if(logging == True):
        print("Clipping method: ", args.clipping)

    print("Mode: DGN (%s), IC (%s), BC (%s), classical(%s), AN( %s)" %  \
          (args.enable_diminishing_gradient_norm, enable_individual_clipping, \
           enable_batch_clipping, enable_classical_BC, \
           args.ci_as_average_norm))

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if(args.shuffle_dataset):
        train_kwargs = {'batch_size': args.batch_size,  'shuffle': True}
    else:
        train_kwargs = {'batch_size': args.batch_size,  'shuffle': False}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}

    if use_cuda:
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True,
                       }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # model = AlexNet(num_classes=10).to(device)
    # model_name = "AlexNet"
    # model = SimpleDLA().to(device)
    # model = nor_convnet(num_classes=10).to(device)
    # model_name = "nor_convnet"
    model = convnet(num_classes=10).to(device)
    model_name = "convnet"
    # model = ResNet18(num_classes=10).to(device)
    # model_name = "resnet18"
    # model = ResNet18_no_BN(num_classes=10).to(device)
    # model_name = "resnet18_no_BN"

    # summary(model,(3, 32, 32))
    # print(model)
    # input()
    # model = PlainNet18(num_classes=10).to(device)
    # model_name = "plainnet18"
    # model = LeNet().to(device)
    # model_name = "LeNet"
    # model = nor_LeNet().to(device)
    # model_name = "nor_LeNet"
    # model = SquareNet().to(device)
    # model_name = "squarenet"
    # number_of_layer = get_number_of_layer(model, model_name)
    # print(number_of_layer)
    # input("HERE")
    if(args.opacus_training):
        # Fix incompatiple components such as BatchNorm2D layer
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)
    if(sigma_discount_on):
        number_of_layer = get_number_of_layer(model,model_name)
        args.noise_multiplier = args.noise_multiplier / number_of_layer

    # model = nor_LeNet().to(device)
    # model_name = "nor_LeNet"

    # model = nor_convnet(num_classes=10).to(device)
    # model_name = "nor_convnet"
    # BNF_nor_convnet_model
    # model = BNF_convnet(num_classes=10).to(device)
    # model_name = "BNF_convnet"
    print("Training with model:", model_name)
    optimizer = get_optimizer(args.optimizer,model ,args.lr)
    """VGG 16 """
    # arch = [64, 64, 'M',
    #         128, 128, 'M',
    #         256, 256, 256, 'M',
    #         512, 512, 512, 'M',
    #         512, 512, 512, 'M']
    # model = VGGNet(in_channels=3, num_classes=10, arch=arch).to(device)
    # model = torchvision.models.vgg16().to(device)
    # input_lastLayer = model.classifier[6].in_features
    # print(input_lastLayer)
    # model.classifier[6] = torch.nn.Linear(input_lastLayer,10)
    # model = model.to(device)
    # input(model)
    # model = VGG16(num_classes=10).to(device)
    # model_name = "VGG16"
    # optimizer = MNIST_optimizer.SGD_optimizer(args.lr,model)
    # sigma = 6
    # gradient_norm = 3
    # optimizer_name = "DPSGD"
    # optimizer_name = "SGD"



    print('Initializing visualization...')
    # visualizer = Visualizer({"name": "MNIST DPSGD"})
    visualizer = None
    train_accuracy = []
    test_accuracy = []
    if(args.enable_DP == True):
        out_file_path = "./graphs/data_neurips/" + settings_file +  "/" + model_name + "/" + args.optimizer + "/" + str(args.clipping)
        if (args.clipping == "layerwise" and args.constant_c_i == True):
            out_file_path = out_file_path + "/constant_c_i"
    else:
        out_file_path = "./graphs/data_neurips/" + settings_file +  "/" + model_name + "/" + args.optimizer
    # Get training and testing data loaders
    # train_batches, test_loader, dataset_size = dataset_preprocessing(args.dataset_name, train_kwargs,
    #                                                                  test_kwargs,
    #                                                                 )

    C_dataset_loader, train_loader, test_loader, dataset_size = dataset_preprocessing(args.dataset_name, train_kwargs,
                                                                     test_kwargs,mode
                                                                     )

    if (args.enable_DP and not args.opacus_training):
        if(args.clipping == "layerwise" or args.clipping == "weight_FGC"):
            if (args.constant_c_i):
                number_of_layer = get_number_of_layer(model,model_name)
                args.each_layer_C = [args.max_grad_norm]*number_of_layer
            else:
                at_epoch = 5
                dummy_model = copy.deepcopy(model)
                dummy_optimizer = get_optimizer(args.optimizer,dummy_model ,args.lr)
                if(args.ci_as_average_norm):
                    args.each_layer_C = compute_layerwise_C_average_norm(C_dataset_loader, dummy_model, at_epoch, device,
                                                                         dummy_optimizer, args.max_grad_norm,True)
                else:
                    args.each_layer_C = compute_layerwise_C(C_dataset_loader, dummy_model, at_epoch, device,
                                                        dummy_optimizer, args.max_grad_norm,True)
        print(args.each_layer_C)
    # DP settings:
    print(args.microbatch_size)
    if args.enable_DP:
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
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.noise_multiplier,
                max_grad_norm=max_grad_norm,
                clipping=clipping,
            )
            print("Opacus")
            out_file_path = out_file_path + "/opacus"
        else:
            if (args.microbatch_size == 1):
                print("Individual clipping")
                out_file_path = out_file_path + "/IC"
            elif(args.microbatch_size == args.batch_size):
                print("Batch clipping")
                out_file_path = out_file_path + "/BC"
            elif(enable_classical_BC):
                out_file_path = out_file_path + "/classical"
            else:
                print("Normal Mode")
                out_file_path = out_file_path + "/NM"
        if(args.enable_diminishing_gradient_norm):
            out_file_path = out_file_path + "/DGN"
        elif(args.ci_as_average_norm):
            out_file_path = out_file_path + "/AN"
        if(sigma_discount_on):
            out_file_path = out_file_path + "/discounted"
    else:
        out_file_path = out_file_path + "/SGD"

    # epochs = math.ceil(args.iterations* args.batch_size / dataset_size)
    epochs = 50 #TODO: remove to calculated based on iterations
    print("Total epochs: %f" % epochs)
    print("Saving data to: %s" % out_file_path)
    # print("Saving data to: %s" % out_file_path)

    grad_array = []
    """TRAINING LOOP"""
    for epoch in range(1, epochs + 1):
        print("epoch %s:" % epoch)
        if args.enable_DP:
            if(args.opacus_training):
                print("Opacus training")
                train_acc, gradient_stats = train_model.train(args, model, device, train_loader, optimizer,epoch)
                train_accuracy.append(train_acc)
            elif(enable_classical_BC):
                print("Classical training")
                train_accuracy.append(train_model.DP_train_classical(args, model, device, train_loader, optimizer))
            elif(args.clipping == "weight_FGC"):
                print("Minibatch training")
                train_accuracy.append(train_model.DP_train_weighted_FGC(args, model, device, train_loader, optimizer))
            else:
                print("Minibatch training")
                train_accuracy.append(train_model.DP_train(args, model, device, train_loader, optimizer))
        else:
            print("SGD training")
            train_acc, gradient_stats = train_model.train(args, model, device, train_loader, optimizer,epoch)
            train_accuracy.append(train_acc)
            if(args.save_gradient):
                grad_array.append(gradient_stats)
            # print("HERE")
            # print(gradient_stats)
        test_accuracy.append(validate_model.test(model, device, test_loader))
        """
        DECREASE C VALUE
        """

        if not args.ci_as_average_norm:
            # args.each_layer_C = compute_layerwise_C_average_norm(C_dataset_loader, dummy_model, at_epoch, device,
            #                                                      dummy_optimizer, args.max_grad_norm,True))
            if(args.enable_DP and args.enable_diminishing_gradient_norm and args.clipping == "layerwise" and not args.opacus_training):
                # if (args.brake_C):
                args.max_grad_norm = args.max_grad_norm * args.C_decay
                # Recompute each layer C
                if (args.constant_c_i):
                    number_of_layer = get_number_of_layer(mode,model_name)
                    args.each_layer_C = [args.max_grad_norm]*number_of_layer
                else:
                    args.each_layer_C = compute_layerwise_C(C_dataset_loader, model, 1, device,
                                                        optimizer, args.max_grad_norm,False)
                print("each_layer_C", args.each_layer_C)
        """
        Update learning rate after each epoch
        """
        if (args.gamma < 1):
            # if(test_accuracy[-1] <= test_accuracy[-2]):
            args.lr = args.lr * args.gamma
            print("current learning rate:", args.lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
    if (args.save_gradient):
        grad_out_file_path = out_file_path + "/grad"
        json_to_file(grad_out_file_path, args.load_setting, grad_array)
    generate_json_data_for_graph(out_file_path, args.load_setting, train_accuracy,test_accuracy)

    if args.save_model:
        torch.save(model.state_dict(), "cifar10_cnn.pt")

if __name__ == '__main__':
    main()