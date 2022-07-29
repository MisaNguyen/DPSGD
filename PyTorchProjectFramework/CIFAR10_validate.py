import argparse
import torch
import torch.nn as nn
import numpy as np

# import torch.nn.functional as F
# from datasets import create_dataset
from utils.progression_bar import progress_bar
# from models import create_model
# import os

"""Performs validation of a specified model.
    
Input params:
    config_file: Either a string with the path to the JSON 
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and 
        model-specific settings.
"""
def accuracy(preds, labels):
    return (preds == labels).mean()
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    test_correct = 0
    total = 0
    top1_acc = []
    losses = []
    # test_accuracy = np.array()
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            # compute output
            output = model(data)
            print(output)
            # compute accuracy
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            acc1 = accuracy(preds, labels)
            top1_acc.append(acc1)
            # compute loss
            loss = nn.CrossEntropyLoss()(output, target)
            losses.append(loss.item())
    # test_loss /= len(test_loader.dataset)
    #
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    # visualizer.plot_current_accuracy(epoch, 100*correct / len(test_loader.dataset))
            print(
                f"\tTesting accuracy:\t"
                # f"Loss: {loss:.6f} "
                # f"Acc@1: {train_correct/total:.6f} "
                f"Loss: {np.mean(losses):.6f} "
                f"Acc@1: {np.mean(top1_acc):.6f} "
            )

    return np.mean(top1_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    validate(args.configfile)
