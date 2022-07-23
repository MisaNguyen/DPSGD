
import torch
import argparse
import json
import math
import numpy as np
from models.Lenet_model import Net
from models.resnet_model import ResNet18,ResNet34,ResNet50,ResNet101,ResNet152
# from models.densenet_model import densenet40_k12_cifar10
# from models.alexnet_model import AlexNet
# from models.alexnet_simple import AlexNet
# from models.simple_dla import SimpleDLA
from models.convnet_model import convnet
from datasets import MNIST_dataset, CIFAR10_dataset
from utils.utils import generate_json_data_for_graph
import MNIST_train, MNIST_validate
import CIFAR10_train, CIFAR10_validate

import torch.optim as optim
from utils.visualizer import Visualizer
from CIFAR10_train_opacus import train
""" OPACUS"""
from opacus import PrivacyEngine
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
                        help='Enable individual clippng mode')
    args = parser.parse_args()

    #Add setting path here
    # settings_file = "settings"
    settings_file = "settings_clipping_exp_cifar10_dpsgd"
    print("Running setting: %s.json" % settings_file)
    if(args.load_setting != ""):
        with open(settings_file +".json", "r") as json_file:
            json_data = json.load(json_file)
            setting_data = json_data[args.load_setting]
            # Loading data
            args.batch_size = setting_data["batch_size"]
            # args.microbatch_size = setting_data["microbatch_size"]
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
            args.max_grad_norm = setting_data["max_grad_norm"]
            args.optimizer = setting_data["optimizer"]
            args.enable_diminishing_gradient_norm = setting_data["diminishing_gradient_norm"]
            args.enable_individual_clipping = setting_data["is_individual_clipping"]
            args.enable_DP = setting_data["enable_DP"]
            args.clip_per_layer = False #TODO: add to setting file
            args.secure_rng = False #TODO: add to setting file
            clipping = "per_layer" if args.clip_per_layer else "flat"
    print("Mode: DGN (%s), IC (%s)" %  (args.enable_diminishing_gradient_norm, args.enable_individual_clipping))
    # if (args.enable_diminishing_gradient_norm == True):
    #     mode = "DGN"
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    # print(args.batch_size)
    train_kwargs = {'batch_size': args.batch_size,  'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}

    # train_loader, test_loader = MNIST_dataset.create_dataset(train_kwargs,test_kwargs)


    if use_cuda:
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True,
                       }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    # input(len(train_loader))
    # input(len(test_kwargs))
    # model = Net().to(device)
    # model = densenet40_k12_cifar10().to(device)
    # model = AlexNet(num_classes=10).to(device)
    # model = SimpleDLA().to(device)
    model = convnet(num_classes=10).to(device)
    model_name = "convnet"
    # optimizer = MNIST_optimizer.SGD_optimizer(args.lr,model)
    # sigma = 6
    # gradient_norm = 3
    # optimizer_name = "DPSGD"
    # optimizer_name = "SGD"

    if args.optimizer == "SGD":
        # optimizer = optim.SGD(
        #     model.parameters(),
        #     lr=args.lr,
        #     # momentum=args.momentum,
        #     # weight_decay=args.weight_decay,
        # )
        optimizer= optim.SGD(
            [
                {"params": model.layer1.parameters(), "lr": args.lr},
                {"params": model.layer2.parameters(),"lr": args.lr},
                {"params": model.layer3.parameters(), "lr": args.lr},
                {"params": model.layer4.parameters(), "lr": args.lr},
            ],
            lr=args.lr,
        )
    elif args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        # print(args.optimizer)
        raise NotImplementedError("Optimizer not recognized. Please check spelling")

    print('Initializing visualization...')
    # visualizer = Visualizer({"name": "MNIST DPSGD"})
    visualizer = None
    train_accuracy = []
    test_accuracy = []
    out_file_path = "./graphs/data/" + settings_file +  "/" + model_name + "/" + args.optimizer
    if args.enable_DP:

        # privacy_engine = None
        if (args.enable_diminishing_gradient_norm == True):
            out_file_path = out_file_path + "/DGN"
        if (args.enable_individual_clipping == True):
            train_loader, test_loader, dataset_size = CIFAR10_dataset.individual_clipping_preprocessing(train_kwargs,test_kwargs)

            privacy_engine = PrivacyEngine(
                secure_mode=args.secure_rng,
            )
            clipping = "per_layer" if args.clip_per_layer else "flat"
            """
            Opacus privacy engine
            """
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.noise_multiplier,
                max_grad_norm=args.max_grad_norm,
                clipping=clipping,
                poisson_sampling=False,
            )
            out_file_path = out_file_path + "/IC"
        else:
            train_batches, test_loader, dataset_size = CIFAR10_dataset.batch_clipping_preprocessing(train_kwargs,test_kwargs)
            out_file_path = out_file_path + "/BC"
    else:
        # train_loader, test_loader, dataset_size = CIFAR10_dataset.minibatch_SGD_preprocessing(train_kwargs,test_kwargs)
        train_loader, test_loader, dataset_size = CIFAR10_dataset.minibatch_SGD_preprocessing(train_kwargs,test_kwargs)

    # epochs = math.ceil(args.iterations* args.batch_size / dataset_size)
    epochs = 50 #TODO: remove to calculated based on iterations
    print("Total epochs: %f" % epochs)
    print("Saving data to: %s" % out_file_path)

    """TRAINING LOOP"""
    for epoch in range(1, epochs + 1):
        print("epoch %s:" % epoch)
        if args.enable_DP:
            if(args.enable_individual_clipping):

                # input(list(train_loader))
                train_accuracy.append(CIFAR10_train.train(args, model, device, train_loader, optimizer,
                                                      args.enable_diminishing_gradient_norm,
                                                      args.enable_individual_clipping))
            else:

                train_accuracy.append(CIFAR10_train.BC_train(args, model, device, train_batches, epoch, optimizer,
                                                      args.enable_diminishing_gradient_norm,
                                                      args.enable_individual_clipping))
        else:


            train_accuracy.append(CIFAR10_train.train(args, model, device, train_loader, optimizer,
                                                         args.enable_diminishing_gradient_norm,
                                                         args.enable_individual_clipping))

        test_accuracy.append(CIFAR10_validate.test(model, device, test_loader))
    generate_json_data_for_graph(out_file_path, args.load_setting, train_accuracy,test_accuracy)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()