import argparse

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
# from datasets import create_dataset
# from utils import parse_configuration
# import math
# from models import create_model
# import time


from optimizers import DPSGD_optimizer
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

    for batch_idx, (data, target) in enumerate(train_loader):
        # visualizer.reset()
        data, target = data.to(device), target.to(device)
        # input(data.shape)
        # input(target.shape)
        if optimizer_name == "DPSGD":
            optimizer = MNIST_optimizer.DPSGD_optimizer(model.parameters(),args.lr,
                                                        args.noise_multiplier,args.max_grad_norm)
        elif optimizer_name == "SGD":
            optimizer = MNIST_optimizer.SGD_optimizer(model.parameters(),args.lr,)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

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
        loss.backward()

        optimizer.step()
        # Decrease step_size after one batch
        scheduler.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            # losses = model.get_current_losses()
        #     visualizer.print_current_losses(epoch, batch_idx * len(data), len(train_loader.dataset), loss)

            # visualizer.plot_current_losses(epoch, batch_idx / len(train_loader), loss)

        if args.dry_run:
            break

def train_sample(args, model, device, train_loader, optimizer_name, epoch):
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        for param in model.parameters():
            param.accumulated_grads = []

        # Run the microbatches
        # for sample_idx, _ in enumerate(data):
            # input(sample)

            # x, y = data[sample_idx], target[sample_idx]
            # y_hat = model(x)
            # torch.no_grad()
        output = model(data)
        # criterion = torch.nn.CrossEntropyLoss
        # loss = criterion(y_hat, y)

        loss = F.nll_loss(output, target)
        # input(output)
        # input(target)
        loss.backward()

        # Clip each parameter's per-sample gradient
        for param in model.parameters():

            per_sample_grad = param.grad.detach().clone()
            torch.nn.utils.clip_grad_norm_(per_sample_grad, max_norm=args.max_grad_norm)  # in-place
            param.accumulated_grads.append(per_sample_grad)
        # print(len(param.accumulated_grads))
        # for i, param in enumerate(model.parameters()):
        #     input(param.accumulated_grads[i].shape)
                # Aggregate back
        # model.to("cpu")
        # input(param.accumulated_grads)
        # input(param.grad)
        for param in model.parameters():
            # for idx, val in enumerate(param.accumulated_grads):
            #     print(val.shape)
            # input(param.accumulated_grads[0].shape)
            # input(param.accumulated_grads[1].shape)
            # param.grad = torch.stack([param.accumulated_grads[0],param.accumulated_grads[1]], dim=0)
            # param.grad = torch.stack(param.accumulated_grads, dim=0)
            param.grad = torch.cat(param.accumulated_grads, dim=0)

        # Now we are ready to update and add noise!
        for param in model.parameters():
            param.grad += torch.normal(mean=torch.Tensor([0.0]),
                                       std=args.noise_multiplier * args.max_grad_norm).to(device=torch.device("cuda:0"))
            param = param - args.lr * param.grad
            # param += torch.normal(mean=torch.Tensor([0.0]), std=args.noise_multiplier * args.max_grad_norm).to(device=torch.device("cuda:0"))

            param.grad = None  # Reset for next iteration
        # optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    train(args.configfile)