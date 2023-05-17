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
import validate_model

# from opt_einsum.contract import contract
from datetime import datetime, timedelta
import copy
""" Schedulers """
from scheduler.learning_rate_scheduler import StepLR
from scheduler.gradient_norm_scheduler import StepGN_normal
# from scheduler.noise_multiplier_scheduler import StepLR
# from torch.optim.lr_scheduler import ReduceLROnPlateau
""" Optimizers """
from optimizers import *
""" Utils"""
from utils.utils import compute_layerwise_C
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

def Compute_S_sens(model, data, target):
    # Noise
    # TODO: define data noise (perturbations) here
    noise = 1
    # data with noise
    S_sens = []
    noisy_data = data # TBD
    output = model(noisy_data)
    loss = nn.CrossEntropyLoss(output,target)
    loss.backward(output)
    noisy_params = model.parameters()
    # data
    output = model(data)
    loss = nn.CrossEntropyLoss(output,target)
    loss.backward(output)
    for noisy_param, params in zip(noisy_params,model.parameters):
        # ref: https://pytorch.org/docs/stable/generated/torch.linalg.norm.html#torch.linalg.norm
        # torch.linalg.norm(A, ord=None, dim=None, keepdim=False, *, out=None, dtype=None)
        LHS_norm = torch.linalg.norm(noisy_param.grad - params.grad)
        RHS_norm = torch.linalg.norm(noise)
        S_sens.append(LHS_norm/RHS_norm)
    return S_sens
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

def calculate_full_gradient_norm(model):
    ### sqrt(a^2+b^2) = A sqrt(c^2+d^2) = B, sqrt( a^2+b^2 + c^2+d^2) = sqrt(A^2 + B^2)
    ### sqrt(a^2+b^2) = X > C
    ### sqrt(a^2+b^2)/X*C = C
    ### sqrt(a^2 C^2/X^2 + b^2 C^2/X^2) = C
    flat_grad = []
    for param in model.parameters():
        if isinstance(param.grad, torch.Tensor):
            layer_grad = param.grad # sample grad
        elif isinstance(param.grad, list):
            layer_grad = torch.cat(param.grad, dim=0) # batch grad
        flat_grad.append(layer_grad)

    each_layer_norm = [flat_grad[i].flatten().norm(2,dim=-1) for i in range(len(flat_grad))] # Get each layer norm
    #print("each_layer_norm", each_layer_norm)
    flat_grad_norm = 0
    for i in range(len(each_layer_norm)):
        flat_grad_norm += pow(each_layer_norm[i],2)
        # print("flat_grad_norm",flat_grad_norm)
    flat_grad_norm = np.sqrt(flat_grad_norm)
    return flat_grad_norm
"""
END OPACUS code
"""
def DP_train_classical(args, model, device, train_loader,optimizer):
    model.train()
    print("Training using %s optimizer" % optimizer.__class__.__name__)
    train_loss = 0
    train_correct = 0
    total = 0
    loss = 0
    # Get optimizer

    iteration = 0
    losses = []
    top1_acc = []

    for batch_idx, (batch_data,batch_target) in enumerate(train_loader): # Batch loop
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
        batch = TensorDataset(batch_data,batch_target)
        # print("micro batch size =", args.microbatch_size) ### args.microbatch_size = 1 => Input each data sample
        micro_train_loader = torch.utils.data.DataLoader(batch, batch_size=1,
                                                         shuffle=True) # Load each data

        """ Classical SGD updates"""
        for sample_idx, (data,target) in enumerate(micro_train_loader):
            optimizer_clone.zero_grad()
            iteration += 1
            data, target = data.to(device), target.to(device)

            # compute output
            output = model_clone(data)
            # compute loss
            loss = nn.CrossEntropyLoss()(output, target)
            losses.append(loss.item())
            # compute gradient
            loss.backward()
            # Gradient Descent step
            optimizer_clone.step()
            #####
        # Computing aH
        for param1, param2 in zip(model.parameters(), model_clone.parameters()):
            param1.grad = torch.sub(param2.data,param1.data).div(args.lr) #aH = (W_m - W_0)/eta

            """
            Batch clipping each "batch"
            """
        if(args.clipping == "layerwise"):
            """
            Clip each layer gradients with args.max_grad_norm
            """
            for param in model.parameters:
                # print("Before clipping, grad_norm =", param.grad.data.norm(2))
                torch.nn.utils.clip_grad_norm_(param, max_norm=args.each_layer_C[layer_idx]) # in-place computation, layerwise clipping

        elif (args.clipping == "all"):
            """
            Clip entire gradients with args.max_grad_norm
            """
            """
            Compute flat list of gradient tensors and its norm
            """
            flat_grad_norm = calculate_full_gradient_norm(model)
            """
            Clip all gradients
            """
            if (flat_grad_norm > args.max_grad_norm):
                for param in model.parameters():
                    param.grad = param.grad / flat_grad_norm * args.max_grad_norm
        else:
            raise ValueError("Invalid clipping mode, available options: all, layerwise")

        # Update model
        for layer_idx, param in enumerate(model.parameters()):

            """
            Add Gaussian noise to gradients
            """
            """--------------STATIC NOISE-----------------"""
            # dist = torch.distributions.normal.Normal(torch.tensor(0.0),
            #                                          torch.tensor((2 * args.noise_multiplier *  args.max_grad_norm)))
            """--------------LAYERWISE NOISE-----------------"""
            if(args.clipping=="layerwise"):
                dist = torch.distributions.normal.Normal(torch.tensor(0.0),
                                                         torch.tensor((2 * args.each_layer_C[layer_idx] *  args.noise_multiplier)))
            elif(args.clipping=="all"):
                dist = torch.distributions.normal.Normal(torch.tensor(0.0),
                                                         torch.tensor((2 * args.max_grad_norm *  args.noise_multiplier)))
            # print(param.grad.shape)
            noise = dist.rsample(param.grad.shape).to(device=device)

            # Compute noisy grad
            param.grad = (param.grad + noise).div(len(micro_train_loader))
            # param.grad = param.grad + noise.div(len(micro_train_loader))

        # Update model with noisy grad
        optimizer.step()
        # if (args.mode == "subsampling"):
        #     indices = np.random.permutation(indices) # Reshuffle indices for new round

        """
        Calculate top 1 acc
        """

        batch_data, batch_target = batch_data.to(device), batch_target.to(device)
        output = model(batch_data)
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = batch_target.detach().cpu().numpy()
        acc1 = accuracy(preds, labels)
        top1_acc.append(acc1)
        if batch_idx % (args.log_interval*len(train_loader)) == 0:
            # train_loss += loss.item()
            # prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            #
            # total += batch_target.size(0)
            #
            # train_correct += np.sum(prediction[1].cpu().numpy() == batch_target.cpu().numpy())
            print(
                f"Loss: {np.mean(losses):.6f} "
                f"Acc@1: {np.mean(top1_acc):.6f} "
            )
        if args.dry_run:
            break
    return np.mean(top1_acc)

def DP_train(args, model, device, train_loader,optimizer):
    model.train()
    print("Training using %s optimizer" % optimizer.__class__.__name__)
    train_loss = 0
    train_correct = 0
    total = 0
    loss = 0
    # Get optimizer

    iteration = 0
    losses = []
    top1_acc = []

    for batch_idx, (batch_data,batch_target) in enumerate(train_loader): # Batch loop
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
        # batch = train_batches[indice]
        batch = TensorDataset(batch_data,batch_target)
        # print("micro batch size =", args.microbatch_size) ### args.microbatch_size = s/m
        micro_train_loader = torch.utils.data.DataLoader(batch, batch_size=args.microbatch_size,
                                                         shuffle=True) # Load each data

        """ Original SGD updates"""
        for sample_idx, (data,target) in enumerate(micro_train_loader):
            optimizer_clone.zero_grad()
            iteration += 1
            data, target = data.to(device), target.to(device)
            # compute output
            output = model_clone(data)
            # compute loss
            loss = nn.CrossEntropyLoss()(output, target)
            losses.append(loss.item())
            # compute gradient
            loss.backward()

            # Add grad to sum of grad
            """
            Batch clipping each "microbatch"
            """
            if(args.clipping == "layerwise"):
                """------------------------------------------------"""
                # for layer_idx, (name, param) in enumerate(model_clone.named_parameters()):
                    # if("test_layer" in name):
                    #     print("layer:", name)
                    #     print("Before clipping, grad_norm =", param.grad.data.norm(2))
                    #     print("Before clipping, grad =", param.grad.data)
                    #     print("Before clipping, c_i =", args.each_layer_C[layer_idx])
                    #     torch.nn.utils.clip_grad_norm_(param, max_norm=args.each_layer_C[layer_idx])
                    #     print("After clipping, grad_norm =", param.grad.data.norm(2))
                    #     print("After clipping, grad =", param.grad.data)
                    #     print("After clipping, c_i =", args.each_layer_C[layer_idx])
                        # input()
                for layer_idx, param in enumerate(model_clone.parameters()):
                    """
                    Clip each layer gradients with args.max_grad_norm
                    """
                    # print("-"*20)
                    # print("Before clipping, grad_norm =", param.grad.data.norm(2))
                    # print("Before clipping, c_i =", args.each_layer_C[layer_idx])
                    # torch.nn.utils.clip_grad_norm_(param.grad, max_norm=args.each_layer_C[layer_idx]) # in-place computation, layerwise clipping
                    torch.nn.utils.clip_grad_norm_(param, max_norm=args.each_layer_C[layer_idx])
                    # print("After clipping, grad_norm =", param.grad.data.norm(2))
                    # print("-"*20)
                    """ 
                    Accumulate gradients
                    """
                    if not hasattr(param, "sum_grad"):
                        param.sum_grad = param.grad
                        # print(param.sum_grad)
                    else:
                        param.sum_grad = param.sum_grad.add(param.grad)
                        # print(param.sum_grad)

            elif (args.clipping == "all"):
                # print("Clipping method: all")
                """
                Clip entire gradients with args.max_grad_norm
                """
                """
                Compute flat list of gradient tensors and its norm 
                """
                flat_grad_norm = calculate_full_gradient_norm(model_clone)
                print("Current norm = ", flat_grad_norm)
                """
                Clip all gradients
                """
                if (flat_grad_norm > args.max_grad_norm):
                    for param in model_clone.parameters():
                        param.grad = param.grad / flat_grad_norm * args.max_grad_norm
                """
                Accumulate gradients
                """
                for param in model_clone.parameters():
                    # param.grad = param.grad / flat_grad_norm * args.max_grad_norm
                    if not hasattr(param, "sum_grad"):
                        param.sum_grad = param.grad
                        # print(param.sum_grad)
                    else:
                        param.sum_grad = param.sum_grad.add(param.grad)
                        # print(param.sum_grad)

            else:
                raise ValueError("Invalid clipping mode, available options: all, layerwise")

        # Copy sum of clipped grad to the model gradient
        for net1, net2 in zip(model.named_parameters(), model_clone.named_parameters()): # (layer_name, value) for each layer
            # input(net2[1])
            net1[1].grad = net2[1].sum_grad
            # print("After copying grad, grad_norm =", net1[1].grad.data.norm(2))
            # input()
            # net1[1].grad = net2[1].sum_grad.div(len(micro_train_loader)) # Averaging the gradients
        # Reset sum_grad
        for param in model_clone.parameters():
            delattr(param, 'sum_grad')
        # Update model
        if(args.noise_multiplier <= 0):
            for layer_idx, (name,param) in enumerate(model.named_parameters()):
                """
                Add Gaussian noise to gradients
                """
                """--------------STATIC NOISE-----------------"""
                # dist = torch.distributions.normal.Normal(torch.tensor(0.0),
                #                                          torch.tensor((2 * args.noise_multiplier *  args.max_grad_norm)))
                """--------------LAYERWISE NOISE-----------------"""
                if(args.clipping=="layerwise"):
                    dist = torch.distributions.normal.Normal(torch.tensor(0.0),
                                                         torch.tensor((2 * args.each_layer_C[layer_idx] *  args.noise_multiplier)))
                elif(args.clipping=="all"):
                    dist = torch.distributions.normal.Normal(torch.tensor(0.0),
                    torch.tensor((2 * args.max_grad_norm * args.noise_multiplier)))
                # print(param.grad.shape)
                # if("test_layer" in name):
                #     print("layer:", name)
                #     # print("Before adding noise, grad_norm =", param.grad.data.norm(2))
                #     print("Before adding noise, grad =", param.grad.data)
                #     print("Before adding noise, c_i =", args.each_layer_C[layer_idx])
                #     noise = dist.rsample(param.grad.shape).to(device=device)
                #     # print("Noise value =", noise)
                #     param.grad = (param.grad + noise).div(len(micro_train_loader))
                    # print("After clipping, grad_norm =", param.grad.data.norm(2))
                    # print("After adding noise, grad =", param.grad.data)
                    # print("After adding noise, c_i =", args.each_layer_C[layer_idx])
                    # input()
                noise = dist.rsample(param.grad.shape).to(device=device)

                # Compute noisy grad
                param.grad = (param.grad + noise).div(len(micro_train_loader))
                # param.grad = param.grad + noise.div(len(micro_train_loader))

        # Update model with noisy grad
        optimizer.step()
        # if (args.mode == "subsampling"):
        #     indices = np.random.permutation(indices) # Reshuffle indices for new round

        """
        Calculate top 1 acc
        """
        # batch = train_batches[indice]
        # input(len(batch))
        batch_data, batch_target = batch_data.to(device), batch_target.to(device)
        output = model(batch_data)
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = batch_target.detach().cpu().numpy()
        acc1 = accuracy(preds, labels)
        top1_acc.append(acc1)
        if batch_idx % (args.log_interval*len(train_loader)) == 0:

            # train_loss += loss.item()
            # prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            #
            # total += batch_target.size(0)
            #
            # train_correct += np.sum(prediction[1].cpu().numpy() == batch_target.cpu().numpy())
            print(
                f"Loss: {np.mean(losses):.6f} "
                f"Acc@1: {np.mean(top1_acc):.6f} "
            )
    return np.mean(top1_acc)


def train(args, model, device, train_loader,
          optimizer,epoch):
    model.train()
    print("Training using %s optimizer" % optimizer.__class__.__name__)

    gradient_stats = {"epoch" : epoch}

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
        # compute accuracy
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()

        acc1 = accuracy(preds, labels)
        top1_acc.append(acc1)

        loss = nn.CrossEntropyLoss()(output, target)
        losses.append(loss.item())
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        if(args.save_gradient):
            for layer_idx, (name, param) in enumerate(model.named_parameters()):
                if param.requires_grad:
                    # count = count + 1
                    # layer_number = "layer_" + str(count)
                    layer_name = "layer_" + str(name)
                    # print(layer_name)
                    if (layer_name in gradient_stats):
                        gradient_stats[layer_name]["norm"].append(float(param.grad.data.norm(2)))
                        gradient_stats[layer_name]["norm_avg"].append(float(param.grad.data.norm(2)/ param.grad.shape[0]))
                    else:
                        gradient_stats[layer_name] = {}
                        gradient_stats[layer_name]["shape"] = param.grad.shape
                        # print(type(param.grad.shape))
                        gradient_stats[layer_name]["norm"] = [float(param.grad.data.norm(2))]
                        gradient_stats[layer_name]["norm_avg"] = [float(param.grad.data.norm(2)/ param.grad.shape[0])]
            """
            {"epoch": 0,
             "Layer_1": {
                 "shape": aaa,
                 "norm": [bbb],
                 "norm_avg": [ccc]},
                 ...
              "Layer_n": {}}
            """

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

    return np.mean(top1_acc), gradient_stats
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    train(args.configfile)
