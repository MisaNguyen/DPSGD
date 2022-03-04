import argparse
import torch
import torch.nn.functional as F
from datasets import create_dataset
from utils import parse_configuration
from models import create_model
import os

"""Performs validation of a specified model.
    
Input params:
    config_file: Either a string with the path to the JSON 
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and 
        model-specific settings.
"""
def test(model, device, test_loader,epoch,visualizer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # input(pred)
            # input(target)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # print(correct)
            # print(len(test_loader.dataset))

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # visualizer.plot_current_accuracy(epoch, 100*correct / len(test_loader.dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    validate(args.configfile)
