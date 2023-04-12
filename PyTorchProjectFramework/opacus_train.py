import warnings
warnings.simplefilter("ignore")
import numpy as np
import argparse
from opacus.utils.batch_memory_manager import BatchMemoryManager
from utils.utils import  json_to_file
from tqdm import tqdm
from models.convnet_model import convnet
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision import models

from opacus.validators import ModuleValidator
from opacus import PrivacyEngine

import torch.nn as nn
import torch.optim as optim



def accuracy(preds, labels):
    return (preds == labels).mean()

def train(model, train_loader, optimizer, epoch, device,
          MAX_PHYSICAL_BATCH_SIZE,privacy_engine,DELTA):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
            optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

            if (i+1) % 200 == 0:
                epsilon = privacy_engine.get_epsilon(DELTA)
                # eps_delta.append([epsilon,DELTA])
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {DELTA})"
                )
    epsilon = privacy_engine.get_epsilon(DELTA)
    # eps_delta.append([epsilon,DELTA])
    return np.mean(top1_acc), [epsilon,DELTA]


def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
    )
    return np.mean(top1_acc)





def main():

    settings = {
        "setting_11": {
            "MAX_GRAD_NORM": 1.0,
            "noise_multiplier": 2.0,
            "EPOCHS": 20,
            "LR": 0.025,
            "BATCH_SIZE": 64,
            "MAX_PHYSICAL_BATCH_SIZE": 64,
            "gamma": 0.9,
            "clipping": "flat"
        },
        "setting_12": {
            "MAX_GRAD_NORM": 1.0,
            "noise_multiplier": 4.0,
            "EPOCHS": 20,
            "LR": 0.025,
            "BATCH_SIZE": 64,
            "MAX_PHYSICAL_BATCH_SIZE": 64,
            "gamma": 0.9,
            "clipping": "flat"
        },
        "setting_13": {
            "MAX_GRAD_NORM": 1.0,
            "noise_multiplier": 8.0,
            "EPOCHS": 20,
            "LR": 0.025,
            "BATCH_SIZE": 64,
            "MAX_PHYSICAL_BATCH_SIZE": 64,
            "gamma": 0.9,
            "clipping": "flat"
        },
        "setting_21": {
            "MAX_GRAD_NORM": 1.0,
            "noise_multiplier": 2.0,
            "EPOCHS": 20,
            "LR": 0.025,
            "BATCH_SIZE": 128,
            "MAX_PHYSICAL_BATCH_SIZE": 64,
            "gamma": 0.9,
            "clipping": "flat"
        },
        "setting_22": {
            "MAX_GRAD_NORM": 1.0,
            "noise_multiplier": 4.0,
            "EPOCHS": 20,
            "LR": 0.025,
            "BATCH_SIZE": 128,
            "MAX_PHYSICAL_BATCH_SIZE": 64,
            "gamma": 0.9,
            "clipping": "flat"
        },
        "setting_23": {
            "MAX_GRAD_NORM": 1.0,
            "noise_multiplier": 8.0,
            "EPOCHS": 20,
            "LR": 0.025,
            "BATCH_SIZE": 128,
            "MAX_PHYSICAL_BATCH_SIZE": 64,
            "gamma": 0.9,
            "clipping": "flat"
        },
        "setting_31": {
            "MAX_GRAD_NORM": 1.0,
            "noise_multiplier": 2.0,
            "EPOCHS": 20,
            "LR": 0.025,
            "BATCH_SIZE": 256,
            "MAX_PHYSICAL_BATCH_SIZE": 64,
            "gamma": 0.9,
            "clipping": "flat"
        },
        "setting_32": {
            "MAX_GRAD_NORM": 1.0,
            "noise_multiplier": 4.0,
            "EPOCHS": 20,
            "LR": 0.025,
            "BATCH_SIZE": 256,
            "MAX_PHYSICAL_BATCH_SIZE": 64,
            "gamma": 0.9,
            "clipping": "flat"
        },
        "setting_33": {
            "MAX_GRAD_NORM": 1.0,
            "noise_multiplier": 8.0,
            "EPOCHS": 20,
            "LR": 0.025,
            "BATCH_SIZE": 256,
            "MAX_PHYSICAL_BATCH_SIZE": 64,
            "gamma": 0.9,
            "clipping": "flat"
        },
        "setting_41": {
            "MAX_GRAD_NORM": 1.0,
            "noise_multiplier": 2.0,
            "EPOCHS": 20,
            "LR": 0.025,
            "BATCH_SIZE": 512,
            "MAX_PHYSICAL_BATCH_SIZE": 64,
            "gamma": 0.9,
            "clipping": "flat"
        },
        "setting_42": {
            "MAX_GRAD_NORM": 1.0,
            "noise_multiplier": 4.0,
            "EPOCHS": 20,
            "LR": 0.025,
            "BATCH_SIZE": 512,
            "MAX_PHYSICAL_BATCH_SIZE": 64,
            "gamma": 0.9,
            "clipping": "flat"
        },
        "setting_43": {
            "MAX_GRAD_NORM": 1.0,
            "noise_multiplier": 8.0,
            "EPOCHS": 20,
            "LR": 0.025,
            "BATCH_SIZE": 512,
            "MAX_PHYSICAL_BATCH_SIZE": 64,
            "gamma": 0.9,
            "clipping": "flat"
        }
    }
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--load-setting', type=str, default="setting_11", metavar='LS',
                        help='Name of setting (example: setting_1, setting_2,...')
    # parser.add_argument('--max-phys-batch-size', type=int, default=64, metavar='N',
    #                     help='input MAX_PHYSICAL_BATCH_SIZE for training (default: 64)')
    # parser.add_argument('--delta', type=float, default=1e-5, metavar='N',
    #                     help='delta value (default: 1e-5)')
    args = parser.parse_args()
    print("Training with setting:",args.load_setting)
    setting_name = args.load_setting
    setting_data = settings[setting_name]
    MAX_GRAD_NORM = setting_data["MAX_GRAD_NORM"]
    # EPSILON = setting_data["EPSILON"]
    # DELTA = setting_data["DELTA"]
    NOISE_MULIPLIER = setting_data["noise_multiplier"]
    EPOCHS = setting_data["EPOCHS"]
    LR = setting_data["LR"]
    BATCH_SIZE = setting_data["BATCH_SIZE"]
    MAX_PHYSICAL_BATCH_SIZE = setting_data["MAX_PHYSICAL_BATCH_SIZE"]
    gamma = setting_data["gamma"]
    clipping = setting_data["clipping"]
    test_accuracy = []
    eps_delta_arr = []

    # These values, specific to the CIFAR10 dataset, are assumed to be known.
    # If necessary, they can be computed with modest privacy budgets.
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
    ])

    "DATA preprocessing"
    DATA_ROOT = '../cifar10'

    train_dataset = CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
    )

    test_dataset = CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # MODEL DEFINITION
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device=",device)
    # model = models.resnet18(num_classes=10)
    # model_name ="resnet18"
    model = convnet(num_classes=10)
    model_name ="convnet"
    model = model.to(device)
    # errors = ModuleValidator.validate(model, strict=False)
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)

    """APPLYING DP"""
    privacy_engine = PrivacyEngine()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR)
    # model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    #     module=model,
    #     optimizer=optimizer,
    #     data_loader=train_loader,
    #     epochs=EPOCHS,
    #     target_epsilon=EPSILON,
    #     target_delta=DELTA,
    #     max_grad_norm=MAX_GRAD_NORM,
    # )
    # num_grad_layers = 0
    num_grad_layers = 62 #resnet18
    # for layer_idx, (name, param) in enumerate(model.named_parameters()):
    #     if param.requires_grad:
    #         num_grad_layers = num_grad_layers + 1
    # print("num_grad_layers=",num_grad_layers)
    new_NOISE_MULIPLIER = NOISE_MULIPLIER / np.power(num_grad_layers,1/4)
    # print("new_NOISE_MULIPLIER=",new_NOISE_MULIPLIER)
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=new_NOISE_MULIPLIER,
        max_grad_norm=MAX_GRAD_NORM,
        clipping=clipping,
    )

    DELTA = 1/len(train_dataset)
    print("delta= ", DELTA)
    print(f"Using sigma={new_NOISE_MULIPLIER} and C={MAX_GRAD_NORM}")
    for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
        train_acc, eps_delta = train(model, train_loader, optimizer, epoch + 1, device,
              MAX_PHYSICAL_BATCH_SIZE,privacy_engine,DELTA)
        eps_delta_arr.append(eps_delta)
        """
        Update learning rate if test_accuracy does not increase
        """
        test_accuracy.append(test(model, test_loader, device))
        if (epoch > 2):
            if(test_accuracy[-1] <= test_accuracy[-2]):
                LR = LR * gamma
                # print(args.lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = LR
    out_json = {
        "test_acc" : test_accuracy,
        "eps_delta": eps_delta,
        "sigma" : NOISE_MULIPLIER,
        "sigma_prime": new_NOISE_MULIPLIER
    }
    out_file_path = "graphs/data_sum_flaw/opacus_" + model_name
    json_to_file(out_file_path, setting_name, out_json)

if __name__ == '__main__':
    main()