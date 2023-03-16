
import torch
import argparse
import json
import copy
from models.resnet_model import ResNet18

"""MODELS"""
# from models.densenet_model import densenet40_k12_cifar10
# from models.alexnet_model import AlexNet
# from models.alexnet_simple import AlexNet
# from models.simple_dla import SimpleDLA
# from models.nor_convnet_model import nor_convnet
# from models.vgg16 import VGGNet
"""DATASETS"""

from datasets.dataset_preprocessing import dataset_preprocessing
"""UTILS"""
from utils.utils import generate_json_data_for_graph, json_to_file
from utils.utils import compute_layerwise_C

"""TRAIN AND VALIDATE"""
import validate_model
import train_model

"""OPTIMIZERS"""
import torch.optim as optim

""" OPACUS"""


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
    Define sampling method here
    """
    enable_individual_clipping = False
    enable_batch_clipping = True
    # mode = "subsampling"
    mode = "shuffling"
    # mode = None
    settings_file = "settings_clipping_exp_cifar10_dpsgd"
    logging = True
    print("Running setting: %s.json" % settings_file)
    if (mode != None):
        settings_file = settings_file + "_" + mode
    # if(args.enable_DP):
    if(enable_individual_clipping):
        settings_file = settings_file + "_IC"
    elif(enable_batch_clipping):
        settings_file = settings_file + "_BC"
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
            args.max_grad_norm = setting_data["max_grad_norm"]
            args.optimizer = setting_data["optimizer"]
            args.enable_diminishing_gradient_norm = setting_data["diminishing_gradient_norm"]
            # args.enable_individual_clipping = setting_data["is_individual_clipping"]
            # args.enable_batch_clipping = False
            # args.enable_DP = setting_data["enable_DP"]
            args.enable_DP = True #TODO: Change here before upload to github
            # args.clip_per_layer = False #TODO: add to setting file
            # args.secure_rng = False #TODO: add to setting file
            args.shuffle_dataset = True
            # args.is_partition_train = False
            args.mode = setting_data["data_sampling"]
            args.clipping = "layerwise"#TODO: add to setting file
            # args.clipping = "all"
            args.C_decay = 0.9
            # args.dataset_name = "MNIST"
            args.dataset_name = "CIFAR10"#TODO: add to setting file


    if(logging == True):
        print("Clipping method: ", args.clipping)

    print("Mode: DGN (%s), IC (%s)" %  (args.enable_diminishing_gradient_norm, args.enable_individual_clipping))

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
    # model = convnet(num_classes=10).to(device)
    # model_name = "convnet"
    model = ResNet18().to(device)
    model_name = "resnet18"
    # model = SquareNet().to(device)
    # model_name = "squarenet"
    # model = LeNet().to(device)
    # model_name = "LeNet"
    # model = nor_LeNet().to(device)
    # model_name = "nor_LeNet"

    # model = nor_convnet(num_classes=10).to(device)
    # model_name = "nor_convnet"
    # BNF_nor_convnet_model
    # model = BNF_convnet(num_classes=10).to(device)
    # model_name = "BNF_convnet"

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
        out_file_path = "./graphs/data_sum/" + settings_file +  "/" + model_name + "/" + args.optimizer + "/" + str(args.clipping)
    else:
        out_file_path = "./graphs/data_sum/" + settings_file +  "/" + model_name + "/" + args.optimizer
    # Get training and testing data loaders
    # train_batches, test_loader, dataset_size = dataset_preprocessing(args.dataset_name, train_kwargs,
    #                                                                  test_kwargs,
    #                                                                 )

    C_dataset_loader, train_loader, test_loader, dataset_size = dataset_preprocessing(args.dataset_name, train_kwargs,
                                                                     test_kwargs,mode
                                                                     )
    if (args.clipping == "layerwise"):
        at_epoch = 5
        dummy_model = copy.deepcopy(model)
        dummy_optimizer = get_optimizer(args.optimizer,dummy_model ,args.lr)

        args.each_layer_C = compute_layerwise_C(C_dataset_loader, dummy_model, at_epoch, device,
                                                dummy_optimizer, args.max_grad_norm,True)
        print(args.each_layer_C)
    # DP settings:
    print(args.microbatch_size)
    if args.enable_DP:
        # privacy_engine = None
        if (args.enable_diminishing_gradient_norm == True):
            out_file_path = out_file_path + "/DGN"
        if (args.microbatch_size == 1):
            print("Individual clipping")
            out_file_path = out_file_path + "/IC"
        elif(args.microbatch_size == args.batch_size):
            print("Batch clipping")
            out_file_path = out_file_path + "/BC"
        else:
            print("Normal Mode")
            out_file_path = out_file_path + "/NM"
    else:
        out_file_path = out_file_path + "/SGD"

    # epochs = math.ceil(args.iterations* args.batch_size / dataset_size)
    epochs = 50 #TODO: remove to calculated based on iterations
    print("Total epochs: %f" % epochs)
    print("Saving data to: %s" % out_file_path)
    save_grad = False
    grad_array = []
    """TRAINING LOOP"""
    for epoch in range(1, epochs + 1):
        print("epoch %s:" % epoch)
        if args.enable_DP:
            train_accuracy.append(train_model.DP_train(args, model, device, train_loader, optimizer))
        else:
            print("SGD training")
            train_acc, gradient_stats = train_model.train(args, model, device, train_loader, optimizer,epoch)
            train_accuracy.append(train_acc)
            if(save_grad):
                grad_array.append(gradient_stats)
            # print("HERE")
            # print(gradient_stats)
        test_accuracy.append(validate_model.test(model, device, test_loader))
    if (save_grad):
        grad_out_file_path = out_file_path + "/grad"
        json_to_file(grad_out_file_path, args.load_setting, grad_array)

    """
    DECREASE C VALUE
    """
    if(args.C_decay < 1 and args.C_decay > 0):
        args.max_grad_norm = args.max_grad_norm * args.C_decay
        # Recompute each layer C
        args.each_layer_C = compute_layerwise_C(C_dataset_loader, model, 1, device,
                                                optimizer, args.max_grad_norm,False)
    """
    Update learning rate if test_accuracy does not increase
    """
    if (epoch > 2):
        if(test_accuracy[-1] <= test_accuracy[-2]):
            args.lr = args.lr * args.gamma
            print(args.lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
    generate_json_data_for_graph(out_file_path, args.load_setting, train_accuracy,test_accuracy)

    if args.save_model:
        torch.save(model.state_dict(), "cifar10_cnn.pt")

if __name__ == '__main__':
    main()