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
"""
END OPACUS code
"""
#
# def DP_train(args, model, device, train_batches, optimizer):
#     model.train()
#     print("Training using %s optimizer" % optimizer.__class__.__name__)
#     train_loss = 0
#     train_correct = 0
#     total = 0
#     loss = 0
#     # Get optimizer
#
#     iteration = 0
#     losses = []
#     top1_acc = []
#     # indices = np.arange(len(train_batches))
#     if (args.mode == "subsampling"):
#         indices = np.arange(len(train_batches))
#         indices = np.random.permutation(indices) # Shuffle indices
#     elif (args.mode == "shuffling"):
#         indices = np.arange(len(train_batches))
#     else:
#         raise Exception("Invalid train mode")
#
#     for batch_idx in range(len(indices)): # Batch loop
#         indice = indices[batch_idx]
#         optimizer.zero_grad()
#         # copy current model
#
#         model_clone = copy.deepcopy(model)
#         optimizer_clone= optim.SGD(model_clone.parameters(),
#                                    # [
#                                    #     {"params": model_clone.layer1.parameters(), "lr": args.lr},
#                                    #     {"params": model_clone.layer2.parameters(),"lr": args.lr},
#                                    #     {"params": model_clone.layer3.parameters(), "lr": args.lr},
#                                    #     {"params": model_clone.layer4.parameters(), "lr": args.lr},
#                                    # ],
#                                    lr=args.lr,
#                                    )
#         batch = train_batches[indice]
#         micro_train_loader = torch.utils.data.DataLoader(batch, batch_size=args.microbatch_size,
#                                                    shuffle=True) # Load each data
#
#         """ Original SGD updates"""
#         for sample_idx, (data,target) in enumerate(micro_train_loader):
#             optimizer_clone.zero_grad()
#             iteration += 1
#             data, target = data.to(device), target.to(device)
#             # print(target)
#             # output = model(data) # input as batch size = 1
#             # loss = nn.CrossEntropyLoss()(output, target)
#             # loss.backward()
#
#             # compute output
#             output = model_clone(data)
#             # compute loss
#             loss = nn.CrossEntropyLoss()(output, target)
#             losses.append(loss.item())
#             # compute gradient
#             loss.backward()
#             # Add grad to sum of grad
#             for param in model_clone.parameters():
#
#                 # """
#                 # Batch clipping each "sample"
#                 # """
#                 # if(args.enable_diminishing_gradient_norm == True):
#                 #     # args.max_grad_norm = torch.linalg.norm(param.grad).to("cpu")
#                 #     # print(args.max_grad_norm)
#                 #     if not hasattr(param, "prev_max_grad_norm"): #round 1
#                 #         torch.nn.utils.clip_grad_norm_(param.grad, max_norm=args.max_grad_norm) # in-place computation
#                 #     else: #round 2 onward
#                 #         torch.nn.utils.clip_grad_norm_(param.grad, max_norm=param.prev_max_grad_norm) # in-place computation
#                 #     if not hasattr(param, "layer_max_grad_norm"):
#                 #         param.layer_max_grad_norm  = torch.linalg.norm(param.grad)
#                 #     else:
#                 #         param.layer_max_grad_norm = max(param.layer_max_grad_norm, torch.linalg.norm(param.grad)) # get new max_grad_norm
#                 #     # print(param.layer_max_grad_norm)
#                 # else:
#                 #     torch.nn.utils.clip_grad_norm_(param.grad, max_norm=args.max_grad_norm) # in-place computation
#
#                 if not hasattr(param, "sum_grad"):
#                     param.sum_grad = param.grad
#                 else:
#                     param.sum_grad = param.sum_grad.add(param.grad)
#
#             # Gradient Descent step
#
#             # optimizer_clone.step()
#
#         # Copy sum of grad to the model gradient
#         for net1, net2 in zip(model.named_parameters(), model_clone.named_parameters()): # (layer_name, value) for each layer
#             # net1[1].grad = net2[1].sum_grad
#             net1[1].grad = net2[1].sum_grad.div(len(micro_train_loader))
#         # Reset sum_grad
#         for param in model_clone.parameters():
#             delattr(param, 'sum_grad')
#         # Update model
#         for param in model.parameters():
#             # param.grad = torch.mul(param.accumulated_grads,1/args.batch_size)
#             # param.grad = param_clone.sum_grad.clone # Copy sum_grad
#             """
#             Batch clipping for each "micro batch"
#             """
#             if(args.enable_diminishing_gradient_norm == True):
#                 # args.max_grad_norm = torch.linalg.norm(param.grad).to("cpu")
#                 # print(args.max_grad_norm)
#                 if not hasattr(param, "prev_max_grad_norm"): #round 1
#                     torch.nn.utils.clip_grad_norm_(param.grad, max_norm=args.max_grad_norm) # in-place computation
#                 else: #round 2 onward
#                     torch.nn.utils.clip_grad_norm_(param.grad, max_norm=param.prev_max_grad_norm) # in-place computation
#                 if not hasattr(param, "layer_max_grad_norm"):
#                     param.layer_max_grad_norm  = torch.linalg.norm(param.grad)
#                 else:
#                     param.layer_max_grad_norm = max(param.layer_max_grad_norm, torch.linalg.norm(param.grad)) # get new max_grad_norm
#                 # print(param.layer_max_grad_norm)
#             else:
#                 torch.nn.utils.clip_grad_norm_(param.grad, max_norm=args.max_grad_norm) # in-place computation
#             """
#             Add Gaussian noise to gradients
#             """
#             dist = torch.distributions.normal.Normal(torch.tensor(0.0),
#                                                      torch.tensor((2 * args.noise_multiplier *  args.max_grad_norm)))
#
#             noise = dist.rsample(param.grad.shape).to(device=device)
#
#             # param.grad = param.grad + noise / args.batch_size
#             # input(param.grad)
#             # param.grad = (param.grad + noise).div(len(micro_train_loader))
#             param.grad = param.grad + noise.div(len(micro_train_loader))
#             # print("----------------------")
#             # input(param.grad)
#
#         optimizer.step()
#         if (args.mode == "subsampling"):
#             indices = np.random.permutation(indices) # Reshuffle indices for new round
#
#         """
#         Calculate top 1 acc
#         """
#         batch = train_batches[indice]
#         # input(len(batch))
#         data_loader = torch.utils.data.DataLoader(batch, batch_size=args.microbatch_size) # Load each data
#         for data, target in data_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             preds = np.argmax(output.detach().cpu().numpy(), axis=1)
#             labels = target.detach().cpu().numpy()
#             acc1 = accuracy(preds, labels)
#             top1_acc.append(acc1)
#             # scheduler.step()
#             # input("HERE")
#             if batch_idx % (args.log_interval*len(indices)) == 0:
#
#                 train_loss += loss.item()
#                 prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
#
#                 total += target.size(0)
#
#                 train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
#                 print(
#                     f"Loss: {np.mean(losses):.6f} "
#                     f"Acc@1: {np.mean(top1_acc):.6f} "
#                 )
#             if args.dry_run:
#                 break
#     return np.mean(top1_acc)
# def train(args, model, device, train_loader, optimizer_name, epoch,
#           visualizer,is_diminishing_gradient_norm, is_individual):
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
    # indices = np.arange(len(train_batches))
    # if (args.mode == "subsampling"):
    #     indices = np.arange(len(train_batches))
    #     indices = np.random.permutation(indices) # Shuffle indices
    # elif (args.mode == "shuffling"):
    #     indices = np.arange(len(train_batches))
    # else:
    #     raise Exception("Invalid train mode")

    for batch_idx, (batch_data,batch_target) in enumerate(train_loader): # Batch loop
        # indice = indices[batch_idx]
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
        # print("micro batch size =", args.microbatch_size)
        micro_train_loader = torch.utils.data.DataLoader(batch, batch_size=args.microbatch_size,
                                                         shuffle=True) # Load each data

        """ Original SGD updates"""
        for sample_idx, (data,target) in enumerate(micro_train_loader):
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
            """
            Batch clipping each "microbatch"
            """
            if(args.clipping == "layerwise"):
                # each_layer_C = []
                # """
                # Layerwise custom C
                # """
                #
                # prev_layer_norm = None
                # for name, param in model_clone.named_parameters():
                #     if param.requires_grad:
                #         layer_name = "layer_" + str(name)
                #         current_layer_norm = param.grad.data.norm(2).clone().detach()
                #
                #         if not each_layer_C:
                #             each_layer_C.append(args.max_grad_norm)
                #         else:
                #             C_ratio = current_layer_norm / prev_layer_norm
                #
                #             each_layer_C.append(each_layer_C[-1]*float(C_ratio))
                #         prev_layer_norm = current_layer_norm

                # print(each_layer_C)

                """------------------------------------------------"""

                for layer_idx, param in enumerate(model_clone.parameters()):

                    """
                    Clip each layer gradients with args.max_grad_norm
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
                        # torch.nn.utils.clip_grad_norm_(param.grad, max_norm=args.max_grad_norm) # in-place computation
                        torch.nn.utils.clip_grad_norm_(param.grad, max_norm=args.each_layer_C[layer_idx]) # in-place computation, layerwise clipping
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
                print("Clipping method: all")
                """
                Clip entire gradients with args.max_grad_norm
                """
                """
                Compute flat list of gradient tensors and its norm 
                """
                flat_grad = []
                for param in model_clone.parameters():
                    if isinstance(param.grad, torch.Tensor):
                        layer_grad = param.grad.cpu() # sample grad
                    elif isinstance(param.grad, list):
                        layer_grad = torch.cat(param.grad, dim=0).cpu() # batch grad
                    flat_grad.append(layer_grad)

                each_layer_norm = [flat_grad[i].flatten().norm(2,dim=-1) for i in range(len(flat_grad))] # Get each layer norm
                flat_grad_norm = 0
                for i in range(len(each_layer_norm)):
                    flat_grad_norm += pow(each_layer_norm[i],2)
                flat_grad_norm = np.sqrt(flat_grad_norm)
                print("Current norm = ", flat_grad_norm)
                # input()
                ### sqrt(a^2+b^2) = A sqrt(c^2+d^2) = B, sqrt( a^2+b^2 + c^2+d^2) = sqrt(A^2 + B^2)
                ### sqrt(a^2+b^2) = X > C
                ### sqrt(a^2+b^2)/X*C = C
                ### sqrt(a^2 C^2/X^2 + b^2 C^2/X^2) = C
                """
                Clip all gradients
                """
                if (flat_grad_norm > args.max_grad_norm):
                    for param in model_clone.parameters():
                        param.grad = param.grad / flat_grad_norm * args.max_grad_norm
                """
                Accumulate gradients
                """
                if not hasattr(param, "sum_grad"):
                    param.sum_grad = param.grad
                    # print(param.sum_grad)
                else:
                    param.sum_grad = param.sum_grad.add(param.grad)
                    # print(param.sum_grad)

                """ ======="""
                # flat_grad = []
                # for param in model_clone.parameters():
                #     if isinstance(param.grad, torch.Tensor):
                #         layer_grad = param.grad # sample grad
                #     elif isinstance(param.grad, list):
                #         layer_grad = torch.cat(param.grad, dim=0) # batch grad
                #     flat_grad.append(layer_grad)

                # each_layer_norm = [flat_grad[i].flatten().norm(2,dim=-1) for i in range(len(flat_grad))] # Get each layer norm
                # flat_grad_norm = 0
                # for i in range(len(each_layer_norm)):
                #     flat_grad_norm += pow(each_layer_norm[i],2)
                # flat_grad_norm = np.sqrt(flat_grad_norm)
                # print("after norm = ", flat_grad_norm)
                # input()
            else:
                raise ValueError("Invalid clipping mode, available options: all, layerwise")

                """
                Clip the entire flat gradients
                """
                # if len(flat_grad[0]) == 0:
                # # Empty batch
                #     per_sample_clip_factor = torch.zeros((0,))
                # else:
                #     per_param_norms = [
                #         g.reshape(len(g), -1).norm(2, dim=-1) for g in flat_grad
                #     ]
                #     for j in range(len(flat_grad)):
                #         print("j=", j)
                #         print(flat_grad[j].shape)
                #         # print(per_param_norms[14].shape)
                #     input()
                #     for i in range(len(per_param_norms)):
                #         print("i=", i)
                #         print(per_param_norms[i].shape)
                #         # print(per_param_norms[14].shape)
                #     input()
                #     per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
                #
                #     per_sample_clip_factor = (
                #             args.max_grad_norm / (per_sample_norms + 1e-6)
                #     ).clamp(max=1.0)
                #     input(per_sample_clip_factor)

                # for param in model_clone.parameters():
                #     # grad = contract("i,i...", per_sample_clip_factor, flat_grad)
                #     if param.summed_grad is not None:
                #         param.summed_grad += grad
                #     else:
                #         param.summed_grad = grad


            # Gradient Descent step

            # optimizer_clone.step()

        # Copy sum of grad to the model gradient
        for net1, net2 in zip(model.named_parameters(), model_clone.named_parameters()): # (layer_name, value) for each layer
            # input(net2[1])
            net1[1].grad = net2[1].sum_grad
            # net1[1].grad = net2[1].sum_grad.div(len(micro_train_loader)) # Averaging the gradients
        # Reset sum_grad
        for param in model_clone.parameters():
            delattr(param, 'sum_grad')
        # Update model
        for layer_idx, param in enumerate(model.parameters()):
            # param.grad = torch.mul(param.accumulated_grads,1/args.batch_size)
            # param.grad = param_clone.sum_grad.clone # Copy sum_grad
            # """
            # Batch clipping for each "micro batch"
            # """
            # if(args.enable_diminishing_gradient_norm == True):
            #     # args.max_grad_norm = torch.linalg.norm(param.grad).to("cpu")
            #     # print(args.max_grad_norm)
            #     if not hasattr(param, "prev_max_grad_norm"): #round 1
            #         torch.nn.utils.clip_grad_norm_(param.grad, max_norm=args.max_grad_norm) # in-place computation
            #     else: #round 2 onward
            #         torch.nn.utils.clip_grad_norm_(param.grad, max_norm=param.prev_max_grad_norm) # in-place computation
            #     if not hasattr(param, "layer_max_grad_norm"):
            #         param.layer_max_grad_norm  = torch.linalg.norm(param.grad)
            #     else:
            #         param.layer_max_grad_norm = max(param.layer_max_grad_norm, torch.linalg.norm(param.grad)) # get new max_grad_norm
            #     # print(param.layer_max_grad_norm)
            # else:
            #     torch.nn.utils.clip_grad_norm_(param.grad, max_norm=args.max_grad_norm) # in-place computation
            """
            Add Gaussian noise to gradients
            """
            """--------------STATIC NOISE-----------------"""
            # dist = torch.distributions.normal.Normal(torch.tensor(0.0),
            #                                          torch.tensor((2 * args.noise_multiplier *  args.max_grad_norm)))
            """--------------LAYERWISE NOISE-----------------"""
            dist = torch.distributions.normal.Normal(torch.tensor(0.0),
                                                     torch.tensor((2 * each_layer_C[layer_idx] *  args.max_grad_norm)))
            noise = dist.rsample(param.grad.shape).to(device=device)

            # param.grad = param.grad + noise / args.batch_size
            # input(param.grad)
            param.grad = (param.grad + noise).div(len(micro_train_loader))
            # param.grad = param.grad + noise.div(len(micro_train_loader))
            # print("----------------------")
            # input(param.grad)

        optimizer.step()
        # if (args.mode == "subsampling"):
        #     indices = np.random.permutation(indices) # Reshuffle indices for new round

        """
        Calculate top 1 acc
        """
        # batch = train_batches[indice]
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
            if batch_idx % (args.log_interval*len(train_loader)) == 0:

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


def train(args, model, device, train_loader,
          optimizer,epoch):
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
        # print(output)
        # compute accuracy
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()

        acc1 = accuracy(preds, labels)
        top1_acc.append(acc1)
        # compute loss
        # previous_loss = loss
        previous_loss = loss
        previous_output = output
        loss = nn.CrossEntropyLoss()(output, target)
        losses.append(loss.item())
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        # count=0
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
            # print("layer_name:", layer_name)
            # print("norm", param.grad.data.norm(2))
            # print("avg_norm:", float(param.grad.data.norm(2)/ param.grad.shape[0]))
            # input()

            """
            {"epoch": 0,
             "Layer_1": {
                 "shape": aaa,
                 "norm": [bbb],
                 "norm_avg": [ccc]},
                 ...
              "Layer_n": {}}
            """
            # print("layer #%d:" % count )
            # print("shape=", param.grad.shape)
            # print("norm= ",param.grad.data.norm(2))
            # print("norm_avg= ",param.grad.data.norm(2)/ param.grad.shape[0])
            # print('-'*20)
        # input()
        if np.isnan(loss.cpu().detach().numpy()):
            print("NaN loss")
            print(batch_idx)
            print(data)
            print(target)
            # imshow(torchvision.utils.make_grid(sample_x.cpu()))
            print(output)
            print("previous loss", previous_loss)
            print("previous output", previous_output)
            # input()
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

    return np.mean(top1_acc), gradient_stats
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    train(args.configfile)
