import argparse
# import numpy as np
# import torch
# import torchvision
# import torch.nn.functional as F
import copy

import torch.nn as nn
# from torch.optim.lr_scheduler import StepLR
# from utils.progression_bar import progress_bar
# from datasets import create_dataset
# from utils import parse_configuration
# import math
# from models import create_model
# import time

import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

from tqdm import tqdm
import CIFAR10_validate

from datetime import datetime, timedelta
import copy
""" Schedulers """
from scheduler.learning_rate_scheduler import StepLR
from scheduler.gradient_norm_scheduler import StepGN_normal
# from scheduler.noise_multiplier_scheduler import StepLR
# from torch.optim.lr_scheduler import ReduceLROnPlateau
""" Optimizers """
from optimizers import *
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



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
"""
OPACUS code
"""

def _get_flat_grad_sample(param: torch.Tensor):
    """
    Return parameter's per sample gradients as a single tensor.
    By default, per sample gradients (``p.grad_sample``) are stored as one tensor per
    batch basis. Therefore, ``p.grad_sample`` is a single tensor if holds results from
    only one batch, and a list of tensors if gradients are accumulated over multiple
    steps. This is done to provide visibility into which sample belongs to which batch,
    and how many batches have been processed.
    This method returns per sample gradients as a single concatenated tensor, regardless
    of how many batches have been accumulated
    Args:
        p: Parameter tensor. Must have ``grad_sample`` attribute
    Returns:
        ``p.grad_sample`` if it's a tensor already, or a single tensor computed by
        concatenating every tensor in ``p.grad_sample`` if it's a list
    Raises:
        ValueError
            If ``p`` is missing ``grad_sample`` attribute
    """

    if not hasattr(param, "grad_sample"):
        raise ValueError(
            "Per sample grad11ient not found. Are you using GradSampleModule?"
        )
    if param.grad_sample is None:
        raise ValueError(
            "Per sample gradient is not initialized. Not updated in backward pass?"
        )
    if isinstance(param.grad_sample, torch.Tensor):
        return param.grad_sample
    elif isinstance(param.grad_sample, list):
        return torch.cat(param.grad_sample, dim=0)
    else:
        raise ValueError(f"Unexpected grad_sample type: {type(param.grad_sample)}")

def grad_samples(params) -> List[torch.Tensor]:
    """
    Returns a flat list of per sample gradient tensors (one per parameter)
    """
    ret = []
    for p in params:
        ret.append(_get_flat_grad_sample(p))
    return ret

def params(optimizer: Optimizer) -> List[nn.Parameter]:
    """
    Return all parameters controlled by the optimizer
    Args:
        optimizer: optimizer
    Returns:
        Flat list of parameters from all ``param_groups``
    """
    ret = []
    for param_group in optimizer.param_groups:
        ret += [p for p in param_group["params"] if p.requires_grad]
    return ret

def accuracy(preds, labels):
    return (preds == labels).mean()
"""
END OPACUS code
"""

def partition_BC_train(args, model, device, train_batches,epoch,
             optimizer):
    model.train()
    print("Training using %s optimizer" % optimizer.__class__.__name__)
    train_loss = 0
    train_correct = 0
    total = 0
    output = 0
    loss = 0
    # Get optimizer
    # train_accuracy = []
    # test_accuracy = []
    iteration = 0
    losses = []
    top1_acc = []
    # Store each layer's max grad norm from last round
    # if(is_diminishing_gradient_norm == True):
    #     for param in model.parameters():
    #         if hasattr(param, "layer_max_grad_norm"):
    #             param.prev_max_grad_norm = param.layer_max_grad_norm
    # for batch_idx, (data,target) in enumerate(train_loader):
    # generate & shuffle batches indices
    indices = np.arange(len(train_batches))
    indices = np.random.permutation(indices)
    for batch_idx, indice in enumerate(tqdm(indices)): # Batch loop
        optimizer.zero_grad()
        # copy current model

        model_clone = copy.deepcopy(model)
        optimizer_clone= optim.SGD(model_clone.parameters(),
                                   # [
                                   #     {"params": model_clone.layer1.parameters(), "lr": args.lr},
                                   #     {"params": model_clone.layer2.parameters(),"lr": args.lr},
                                   #     {"params": model_clone.layer3.parameters(), "lr": args.lr},
                                   #     {"params": model_clone.layer4.parameters(), "lr": args.lr},
                                   # ],
                                   lr=args.lr,
                                   )
        batch = train_batches[indice]
        train_loader = torch.utils.data.DataLoader(batch, batch_size=1, shuffle=True) # Load each data

        """ Original SGD updates"""
        for sample_idx, (data,target) in enumerate(train_loader):
            optimizer_clone.zero_grad()
            iteration += 1
            data, target = data.to(device), target.to(device)
            # print(target)
            # output = model(data) # input as batch size = 1
            # loss = nn.CrossEntropyLoss()(output, target)
            # loss.backward()

            # compute output
            output = model_clone(data)
            # compute loss
            loss = nn.CrossEntropyLoss()(output, target)
            losses.append(loss.item())
            # compute gradient
            loss.backward()
            # Add grad to sum of grad
            for param in model_clone.parameters():
                if not hasattr(param, "sum_grad"):
                    param.sum_grad = param.grad
                else:
                    param.sum_grad = param.sum_grad + param.grad

            # Gradient Descent step
            optimizer_clone.step()

        # Copy sum of grad to the model gradient
        for net1, net2 in zip(model.named_parameters(), model_clone.named_parameters()): # (layer_name, value) for each layer
            net1[1].grad = net2[1].sum_grad
        # Reset sum_grad
        for param in model_clone.parameters():
            delattr(param, 'sum_grad')
        # Update model
        for param in model.parameters():
            # param.grad = torch.mul(param.accumulated_grads,1/args.batch_size)
            # param.grad = param_clone.sum_grad.clone # Copy sum_grad

            """
            Batch clipping
            """
            if(args.enable_diminishing_gradient_norm == True):
                # args.max_grad_norm = torch.linalg.norm(param.grad).to("cpu")
                # print(args.max_grad_norm)
                if not hasattr(param, "prev_max_grad_norm"): #round 1
                    torch.nn.utils.clip_grad_norm_(param.grad, max_norm=args.max_grad_norm) # in-place computation
                else: #round 2 onward
                    torch.nn.utils.clip_grad_norm_(param.grad, max_norm=param.prev_max_grad_norm) # in-place computation
                if not hasattr(param, "layer_max_grad_norm"):
                    param.layer_max_grad_norm  = torch.linalg.norm(param.grad)
                else:
                    param.layer_max_grad_norm = max(param.layer_max_grad_norm, torch.linalg.norm(param.grad)) # get new max_grad_norm
                # print(param.layer_max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(param.grad, max_norm=args.max_grad_norm) # in-place computation

            """
            Add Gaussian noise to gradients
            """
            dist = torch.distributions.normal.Normal(torch.tensor(0.0),
                                                     torch.tensor((args.noise_multiplier *  args.max_grad_norm)))

            noise = dist.rsample(param.grad.shape).to(device=device)

            # param.grad = param.grad + noise / args.batch_size
            # input(param.grad)
            param.grad = (param.grad + noise).div(len(train_loader))
            # print("----------------------")
            # input(param.grad)

        optimizer.step()

        """
        Calculate top 1 acc
        """
        batch = train_batches[indice]
        # input(len(batch))
        data_loader = torch.utils.data.DataLoader(batch, batch_size=args.microbatch_size) # Load each data
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)
            top1_acc.append(acc1)
            # scheduler.step()
            # input("HERE")
            if batch_idx % (args.log_interval*len(indices)) == 0:

                train_loss += loss.item()
                prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced

                total += target.size(0)

                train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
                print(
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc):.6f} "
                )
            if args.dry_run:
                break
    return np.mean(top1_acc)
# def train(args, model, device, train_loader, optimizer_name, epoch,
#           visualizer,is_diminishing_gradient_norm, is_individual):
def BC_train(args, model, device, train_loader,epoch,
          optimizer):
    model.train()
    print("Training using %s optimizer" % optimizer.__class__.__name__)
    train_loss = 0
    train_correct = 0
    total = 0
    output = 0
    loss = 0
    # Get optimizer
    iteration = 0
    losses = []
    top1_acc = []

    for batch_idx, (data,target) in enumerate(tqdm(train_loader)): # Batch loop
        data, target = data.to(device), target.to(device)
        print(data.shape)
        optimizer.zero_grad()
        # copy current model

        model_clone = copy.deepcopy(model)
        optimizer_clone= optim.SGD(model_clone.parameters(),
            # [
            #     {"params": model_clone.layer1.parameters(), "lr": args.lr},
            #     {"params": model_clone.layer2.parameters(),"lr": args.lr},
            #     {"params": model_clone.layer3.parameters(), "lr": args.lr},
            #     {"params": model_clone.layer4.parameters(), "lr": args.lr},
            # ],
            lr=args.lr,
        )
        BatchData = TensorDataset(data,target)
        mini_train_loader = torch.utils.data.DataLoader(BatchData, batch_size=args.microbatch_size, shuffle=True) # Load each data

        """ Original SGD updates"""
        for sample_idx, (micro_data,micro_target) in enumerate(mini_train_loader):
            optimizer_clone.zero_grad()
            iteration += 1
            micro_data, micro_target = micro_data.to(device), micro_target.to(device)
            print(micro_data.shape)
            # print(target)
            # output = model(data) # input as batch size = 1
            # loss = nn.CrossEntropyLoss()(output, target)
            # loss.backward()

            # compute output
            output = model_clone(micro_data)
            # compute loss
            loss = nn.CrossEntropyLoss()(output, micro_target)
            losses.append(loss.item())
            # compute gradient
            loss.backward()
            # Add grad to sum of grad
            for param in model_clone.parameters():
                if not hasattr(param, "sum_grad"):
                    param.sum_grad = param.grad
                else:
                    param.sum_grad = param.sum_grad + param.grad

            # Gradient Descent step
            optimizer_clone.step()

        # Copy sum of grad to the model gradient
        for net1, net2 in zip(model.named_parameters(), model_clone.named_parameters()): # (layer_name, value) for each layer
            net1[1].grad = net2[1].sum_grad
        # Reset sum_grad
        for param in model_clone.parameters():
            delattr(param, 'sum_grad')
        # Update model
        for param in model.parameters():
            # param.grad = torch.mul(param.accumulated_grads,1/args.batch_size)
            # param.grad = param_clone.sum_grad.clone # Copy sum_grad

            """
            Batch clipping
            """
            if(args.enable_diminishing_gradient_norm == True):
                # args.max_grad_norm = torch.linalg.norm(param.grad).to("cpu")
                # print(args.max_grad_norm)
                if not hasattr(param, "prev_max_grad_norm"): #round 1
                    torch.nn.utils.clip_grad_norm_(param.grad, max_norm=args.max_grad_norm) # in-place computation
                else: #round 2 onward
                    torch.nn.utils.clip_grad_norm_(param.grad, max_norm=param.prev_max_grad_norm) # in-place computation
                if not hasattr(param, "layer_max_grad_norm"):
                    param.layer_max_grad_norm  = torch.linalg.norm(param.grad)
                else:
                    param.layer_max_grad_norm = max(param.layer_max_grad_norm, torch.linalg.norm(param.grad)) # get new max_grad_norm
                # print(param.layer_max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(param.grad, max_norm=args.max_grad_norm) # in-place computation

            """
            Add Gaussian noise to gradients
            """
            dist = torch.distributions.normal.Normal(torch.tensor(0.0),
                                                 torch.tensor((args.noise_multiplier * args.max_grad_norm)))

            noise = dist.rsample(param.grad.shape).to(device=device)

            # param.grad = param.grad + noise / args.batch_size
            # input(param.grad)
            param.grad = (param.grad + noise).div(len(train_loader))
            # print("----------------------")
            # input(param.grad)

        optimizer.step()

        """
        Calculate top 1 acc
        """
        output = model(data)
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()
        acc1 = accuracy(preds, labels)
        top1_acc.append(acc1)
        # scheduler.step()
        # input("HERE")
        if batch_idx % args.log_interval == 0:

            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced

            total += target.size(0)

            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
            print(
                f"Loss: {np.mean(losses):.6f} "
                f"Acc@1: {np.mean(top1_acc):.6f} "
            )
        if args.dry_run:
            break
        # ### UPDATE LEARNING RATE after each batch"""
        # if(args.enable_diminishing_gradient_norm):
        #
        #
        #     iterations_per_epoch = len(train_loader)
        # # layer_names = []
        # # # print(len(optimizer.param_groups))
        # # # print(len(model.named_parameters()))
        # # # input()
        # # # for idxparam in model.parameters():
        # # #     print(param)
        # # # for param_group in optimizer.param_groups:
        # # #     print(param_group['lr'])
        # # # for param_group in optimizer.param_groups:
        # # #
        # # #     param_group["lr"] = np.sqrt(iterations_per_epoch)*param_group["param"].layer_max_grad_norm
        # # parameters = []
        # # for idx, (name, param) in enumerate(model.named_parameters()):
        # #     layer_names.append(name)
        # #     parameters+= [{'params': param,
        # #                    'lr':     np.sqrt(iterations_per_epoch)*param.layer_max_grad_norm}]
        # #     print(f'{idx}: lr = {np.sqrt(iterations_per_epoch)*param.layer_max_grad_norm:.6f}, {name}')
        # # optimizer = optim.SGD(parameters)
        # else:
        #     args.lr = args.lr*pow(args.gamma,(epoch-1)*len(train_batches) + batch_idx)
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = param_group["lr"] * args.gamma
    return np.mean(top1_acc)


def train(args, model, device, train_loader,
          optimizer):
    model.train()
    print("Training using %s optimizer" % optimizer.__class__.__name__)
    # train_loss = 0
    # train_correct = 0
    # total = 0
    # output = 0
    # loss = 0
    # Get optimizer
    # train_accuracy = []
    # test_accuracy = []
    iteration = 0
    losses = []
    top1_acc = []
    loss = None
    for batch_idx, (data,target) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        iteration += 1
        data, target = data.to(device), target.to(device)

        # compute output
        output = model(data)
        # print(output)
        # compute accuracy
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()

        acc1 = accuracy(preds, labels)
        top1_acc.append(acc1)
        # compute loss
        previous_loss = loss
        previous_output = output
        loss = nn.CrossEntropyLoss()(output, target)
        losses.append(loss.item())
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        if np.isnan(loss.cpu().detach().numpy()):
            print("NaN loss")
            print(batch_idx)
            print(data)
            print(target)
            # imshow(torchvision.utils.make_grid(sample_x.cpu()))
            print(output)
            print("previous loss", previous_loss)
            print("previous output", previous_output)
            input()
            for param in model.parameters():
                print(param.grad)
        # ### UPDATE LEARNING RATE """
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = param_group["lr"] * args.gamma
            # print(param_group["lr"])
            # input()
        # scheduler.step()
        # input("HERE")
        if batch_idx % (args.log_interval*len(train_loader)) == 0:

            print(
                # f"\tTrain Epoch: {epoch} \t"
                # f"Loss: {loss:.6f} "
                # f"Acc@1: {train_correct/total:.6f} "
                f"Loss: {np.mean(losses):.6f} "
                f"Acc@1: {np.mean(top1_acc):.6f} "
            )
        if args.dry_run:
            break

    return np.mean(top1_acc)
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    train(args.configfile)
