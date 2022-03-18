import argparse
import numpy as np
import torch
# import torch.nn.functional as F
import torch.nn as nn
# from torch.optim.lr_scheduler import StepLR
# from utils.progression_bar import progress_bar
# from datasets import create_dataset
# from utils import parse_configuration
# import math
# from models import create_model
# import time


""" Schedulers """
from scheduler.learning_rate_scheduler import StepLR
from scheduler.gradient_norm_scheduler import StepGN_normal
# from scheduler.noise_multiplier_scheduler import StepLR

""" Optimizers """
from optimizers import MNIST_optimizer
"""Create learning_rate sequence generator
    
Input params:
    decay: learning rate decay
    lr: base learning rate
    epoch: number of training epoch
export: Learning rate generator

"""
def Lr_generator(decay,lr,epoch):
    lr_sequence = range(epoch)
    for index in lr_sequence:
        yield lr*pow(decay,index)


"""Create sample_rate sequence generator
    
Input params:
    multi: sample rate multiplier
    sample_rate: base sample rate
    epoch: number of training epoch
export: sample rate generator

"""
def sample_rate_generator(multi,lr,epoch):
    sr_sequence = range(epoch)
    for index in sr_sequence:
        yield lr*pow(multi,index)


"""Create sigma sequence generator
    
Input params:
    multi: sample rate multiplier
    sample_rate: base sample rate
    epoch: number of training epoch
export: sample rate generator

"""
# def sample_rate_generator(multi,lr,epoch):
#     sr_sequence = range(epoch)
#     for index in sr_sequence:
#         yield lr*pow(multi,index)
"""Performs training of a specified model.
    
Input params:
    config_file: Either a string with the path to the JSON 
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and 
        model-specific settings.
    export: Whether to export the final model (default=True).
"""

def train(args, model, device, train_loader, optimizer_name, epoch,visualizer):
    model.train()
    train_loss = 0
    train_correct = 0
    total = 0
    # Get optimizer
    if optimizer_name == "DPSGD":
        optimizer = MNIST_optimizer.DPSGD_optimizer(model.parameters(),args.lr,
                                                    args.noise_multiplier,args.max_grad_norm)
    elif optimizer_name == "SGD":
        optimizer = MNIST_optimizer.SGD_optimizer(model.parameters(),args.lr,)

    # train_accuracy = np.array()
    # for batch_idx, (data, target) in enumerate(train_loader):
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if optimizer_name == "DPSGD":
            optimizer = MNIST_optimizer.DPSGD_optimizer(model.parameters(),args.lr,
                                                        args.noise_multiplier,args.max_grad_norm)
            for param in model.parameters():
                param.accumulated_grads = []
            # input(len(batch[0]))
            # input(len(batch[1]))
            for sample_idx in range(0,len(data)):
                sample_x, sample_y = data[sample_idx],target[sample_idx]
            # sample_y = target[sample_idx]

                optimizer.zero_grad()
                # Calculate the loss
                output = model(sample_x[None, ...]) # input as batch size = 1
                # input(output.shape)
                # input(target.shape)
                # input(sample_y)
                loss = nn.CrossEntropyLoss()(output, sample_y[None, ...])
                # Loss back-propagation
                loss.backward()
                # train_loss += loss.item()
                # prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
                # total += target.size(0)
                #
                # # train_correct incremented by one if predicted right
                # train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
                #
                # progress_bar(batch_idx, len(train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                #              % (train_loss / (batch_idx + 1), 100. * train_correct / total, train_correct, total))
                # Clip each parameter's per-sample gradient
                for param in model.parameters():
                    per_sample_grad = param.grad.detach().clone()
                    torch.nn.utils.clip_grad_norm_(per_sample_grad, max_norm=args.max_grad_norm)  # in-place
                    param.accumulated_grads.append(per_sample_grad)
                    # print(torch.stack(param.accumulated_grads, dim=0).shape)


            # Aggregate gradients
            # model.to("cpu")
            for param in model.parameters():
                # input(len(param.accumulated_grads))
                accumulated_grads = torch.stack(param.accumulated_grads, dim=0).sum(dim=0)
                # input(param.grad.shape)
                # print(accumulated_grads)
                # input(param.size())
                # input(param.grad.size())
                # input(accumulated_grads.sum(dim=0).size())

                param.grad = accumulated_grads
            # model.to(device)
        elif optimizer_name == "SGD":
            optimizer = MNIST_optimizer.SGD_optimizer(model.parameters(),args.lr)
            optimizer.zero_grad()
            # Calculate the loss
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            # Loss back-propagation
            loss.backward()

        # Get scheduler
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        # Calculate gradient step
        optimizer.step()
        # Decrease learning rate using scheduler
        scheduler.step()

        # Trainning Log
        # output = model(data)
        # loss = nn.CrossEntropyLoss()(output, target)
        # train_loss += loss.item()
        # prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
        # total += target.size(0)
        #
        # # train_correct incremented by one if predicted right
        # train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
        #
        # progress_bar(batch_idx, len(train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1), 100. * train_correct / total, train_correct, total))


        # data, target = data.to(device), target.to(device)
        # print("here",data)
        # input(data.shape)
        # input(data.shape)
        # input(target.shape)



        # optimizer.zero_grad()
        # output = model(data)
        # loss = nn.CrossEntropyLoss()(output, target)
        # loss = F.nll_loss(output, target)

        # # Mini batch
        # for sample in zip(data, target):
        #     x, y = sample
        #     input(x.shape)
        #     input(x)
        #     input(y.shape)
        #     input(y)
        #     optimizer.zero_grad()
        #     y_pred = model(x)
        #     input(y_pred)
        #     # input(output)
        #     # input(target)
        #     loss = F.nll_loss(y_pred, y)

        # loss_object = torch.nn.NLLLoss()
        # lsoftmax = torch.nn.LogSoftmax(dim=-1)
        # loss = torch.nn.NLLLoss()(lsoftmax(output), target)
        #TODO: add optimizer, sample_size_sequence, batch_size_sequence
        # optimizer = MNIST_optimizer.DPSGD_
        # optimizer(model.parameters(),args.lr,sigma,gradient_norm)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            # Trainning Log
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
            # train_accuracy.append(train_correct / total)
            # losses = model.get_current_losses()
        #     visualizer.print_current_losses(epoch, batch_idx * len(data), len(train_loader.dataset), loss)

        # visualizer.plot_current_losses(epoch, batch_idx / len(train_loader), loss)

        if args.dry_run:
            break
    return train_correct / total

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    train(args.configfile)
