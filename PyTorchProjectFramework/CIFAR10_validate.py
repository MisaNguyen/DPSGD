import argparse
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from datasets import create_dataset
from utils.progression_bar import progress_bar
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
    test_correct = 0
    total = 0
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss  = nn.CrossEntropyLoss()(output, target)

            test_loss += loss.item()
            prediction = torch.max(output, 1)
            print("pred:", prediction[1])
            # print("target:", target)
            # print("pred: %f, target: %f" % (prediction[1],target))
            total += target.size(0)
            test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            # input(pred)
            # input(target)
            # correct += pred.eq(target.view_as(pred)).sum().item()
            # print(correct)
            # print(len(test_loader.dataset))
            progress_bar(batch_num, len(test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))
    # test_loss /= len(test_loader.dataset)
    #
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    # visualizer.plot_current_accuracy(epoch, 100*correct / len(test_loader.dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    validate(args.configfile)
