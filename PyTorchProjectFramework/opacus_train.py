import warnings
warnings.simplefilter("ignore")

from utils.utils import  json_to_file

settings = {
    "setting_1": {
        "MAX_GRAD_NORM": 1.2,
        "noise_multiplier": 1.0,
        "EPOCHS": 20,
        "LR": 0.025,
        "BATCH_SIZE": 64,
        "MAX_PHYSICAL_BATCH_SIZE": 64,
        "gamma": 0.9,
        "clipping": "flat"
    },
    "setting_2": {
        "MAX_GRAD_NORM": 0.18,
        "noise_multiplier": 2,
        "EPOCHS": 20,
        "LR": 0.025,
        "BATCH_SIZE": 64,
        "MAX_PHYSICAL_BATCH_SIZE": 64,
        "gamma": 0.9,
        "clipping": "flat"
    },
    "setting_2": {
        "MAX_GRAD_NORM": 1.2,
        "noise_multiplier": 0.5,
        "EPOCHS": 20,
        "LR": 0.025,
        "BATCH_SIZE": 64,
        "MAX_PHYSICAL_BATCH_SIZE": 64,
        "gamma": 0.9,
        "clipping": "flat"
    }
}
setting_name = "setting_2"
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
eps_delta = []
import torch
import torchvision
import torchvision.transforms as transforms

# These values, specific to the CIFAR10 dataset, are assumed to be known.
# If necessary, they can be computed with modest privacy budgets.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
])

from torchvision.datasets import CIFAR10

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

from torchvision import models

model = models.resnet18(num_classes=10)


from opacus.validators import ModuleValidator

errors = ModuleValidator.validate(model, strict=False)
errors[-5:]


model = ModuleValidator.fix(model)
ModuleValidator.validate(model, strict=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device=",device)
model = model.to(device)


import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

def accuracy(preds, labels):
    return (preds == labels).mean()


from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()

# model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
#     module=model,
#     optimizer=optimizer,
#     data_loader=train_loader,
#     epochs=EPOCHS,
#     target_epsilon=EPSILON,
#     target_delta=DELTA,
#     max_grad_norm=MAX_GRAD_NORM,
# )
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=NOISE_MULIPLIER,
    max_grad_norm=MAX_GRAD_NORM,
    clipping=clipping,
)
DELTA = 1/len(train_dataset)
print("delta= ", DELTA)
print(f"Using epsilon={privacy_engine.get_epsilon(1/DELTA)} and C={MAX_GRAD_NORM}")


import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager


def train(model, train_loader, optimizer, epoch, device):
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
                eps_delta.append([epsilon,DELTA])
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {DELTA})"
                )
    return np.mean(top1_acc)


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


from tqdm import tqdm

for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
    train(model, train_loader, optimizer, epoch + 1, device)
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
    "eps_delta": eps_delta
}
out_file_path = "./graphs/data_sum/opacus"
json_to_file(out_file_path, setting_name, out_json)