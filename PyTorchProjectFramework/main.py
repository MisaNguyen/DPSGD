
import torch
import argparse
import json

from models.Lenet_model import Net
from models.resnet_model import ResNet18,ResNet34,ResNet50,ResNet101,ResNet152
# from models.densenet_model import densenet40_k12_cifar10
# from models.alexnet_model import AlexNet
# from models.alexnet_simple import AlexNet
from models.simple_dla import SimpleDLA
from datasets import MNIST_dataset, CIFAR10_dataset
from utils.utils import generate_json_data_for_graph
import MNIST_train, MNIST_validate
import CIFAR10_train, CIFAR10_validate

from utils.visualizer import Visualizer

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
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
    settings_file = "settings_lr_exp_cifar10_dpsgd"
    print("Running setting: %s.json" % settings_file)
    if(args.load_setting != ""):
        with open(settings_file +".json", "r") as json_file:
            json_data = json.load(json_file)
            setting_data = json_data[args.load_setting]
            # Loading data
            args.batch_size = setting_data["batch_size"]
            args.test_batch_size = setting_data["test_batch_size"]
            args.epochs = setting_data["epochs"]
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
    print("Mode: DGN (%s), IC (%s)" %  (args.enable_diminishing_gradient_norm, args.enable_individual_clipping))
    # if (args.enable_diminishing_gradient_norm == True):
    #     mode = "DGN"
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    # TODO:
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    # train_loader, test_loader = MNIST_dataset.create_dataset(train_kwargs,test_kwargs)
    train_loader, test_loader = CIFAR10_dataset.create_dataset(train_kwargs,test_kwargs)
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    # input(len(train_loader))
    # input(len(test_kwargs))
    # model = Net().to(device)
    # model = densenet40_k12_cifar10().to(device)
    # model = AlexNet(num_classes=10).to(device)
    model = SimpleDLA().to(device)
    # optimizer = MNIST_optimizer.SGD_optimizer(args.lr,model)
    sigma = 6
    gradient_norm = 3
    # optimizer_name = "DPSGD"
    # optimizer_name = "SGD"

    print('Initializing visualization...')
    # visualizer = Visualizer({"name": "MNIST DPSGD"})
    visualizer = None
    train_accuracy = []
    test_accuracy = []
    out_file_path = "./graphs/data/" + settings_file + "/" + args.optimizer
    if (args.enable_diminishing_gradient_norm == True):
        out_file_path = out_file_path + "/DGN"
    if (args.enable_individual_clipping == True):
        out_file_path = out_file_path + "/IC"
    print("Saving data to: %s" % out_file_path)
    for epoch in range(1, args.epochs + 1):
        print("epoch %s:" % epoch)
        train_accuracy.append(CIFAR10_train.train(args, model, device, train_loader, args.optimizer, epoch,visualizer,
                                                  args.enable_diminishing_gradient_norm,
                                                  args.enable_individual_clipping))
        test_accuracy.append(CIFAR10_validate.test(model, device, test_loader,epoch,visualizer))
    generate_json_data_for_graph(out_file_path, args.load_setting, train_accuracy,test_accuracy)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()