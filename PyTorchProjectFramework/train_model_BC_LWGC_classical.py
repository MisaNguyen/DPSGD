import pickle

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from collections import OrderedDict
from collections import defaultdict
from torch.func import functional_call, vmap, grad
from torch.utils.data import TensorDataset, DataLoader
"""OPTIMIZERS"""
import torch.optim as optim
import copy

def compute_loss(params, buffers, model, loss_fn, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)

    predictions = functional_call(model, (params, buffers), (batch,))
    loss = loss_fn(predictions, targets)
    return loss

def compute_gradients(model,loss_fn,samples,targets):
    '''
        We want to follow the tutorial in here to compute multiple grads in parallel:
                #https://pytorch.org/tutorials/intermediate/per_sample_grads.html?utm_source=whats_new_tutorials&utm_medium=per_sample_grads
                #https://pytorch.org/docs/stable/generated/torch.func.grad.html
                #https://towardsdatascience.com/introduction-to-functional-pytorch-b5bf739e1e6e
                #https://pytorch.org/functorch/stable/notebooks/per_sample_grads.html [PAY ATTENTION TO]
            Typically, we generate all gradients gis of samples sis in parallel in the helper function: compute_gradients
            The output of compute_gradients is an array called samples_grads
                sample s[0]: samples_grads[0][layer_1], samples_grads[0][layer_2], .... //g0
                    ...............
                sample s[L-1]: samples_grads[L-1][layer_1], samples_grads[L-1][layer_2], ....//g[L-1]
                where L is the number of samples in the mini-batch

            The compute_gradients call another helper function called compute_loss. This is used for computing the gradients in parallel
    '''

    ft_compute_grad = grad(compute_loss)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, None, None, 0, 0))

    '''
    The ft_compute_grad function computes the gradient for a single (sample, target) pair.
    We can use vmap to get it to compute the gradient over an entire batch of samples and targets.
    Note that in_dims=(None, None, 0, 0) because we wish to
    map ft_compute_grad over the 0th dimension of the data and targets, and use the same params and buffers for each.
    '''

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    ft_per_sample_grads = ft_compute_sample_grad(params, buffers,model, loss_fn,samples,targets)

    '''
    ft_per_sample_grads contains the STACKED gradients per layer.
    For example, we have two samples s0 and s1 and we have only two layers "bias" and "weight"
        s0 = ("weight": 1, "layer": 2)
        s1 = ("weight": 3, "layer": 4)
    Stacked gradients per layer means  = ("weight": [1,3], "bias":[2,4])
    Therefore, we have to unstack this stacked gradients to get back the gradient for each sample
    '''

    #get back per_sample_grad
    num_samples = len(samples)
    samples_grads = dict()

    for i in range(num_samples):
      samples_grads[i] = OrderedDict()
      samples_grads[i]['whole_grad'] = list()

    '''
    1. Going through each layer in ft_per_sample_grads: key, value in ft_per_sample_grads.items()
    2. unstack the stacked of len(x) layers: unstacked_grads = torch.unbind(value, dim=0)
    3. redistribute the unstacked sample_layer_grad, i.e., samples_grads[i][key]
    4. We create a new feature called "whole_grad" to combine all layer grads as a whole grad tensor. This is used for computing the full grad norm.
        This full grad norm is used to compute the clipped sample grad later !!!!

    Each sample has its own grad now but saved in the form of dictionary
    '''

    for key,value in ft_per_sample_grads.items():
        #unstack the grads for each layer
        unstacked_grads = torch.unbind(value, dim=0)
        i = 0
        for layer_grad in unstacked_grads:
            samples_grads[i]['whole_grad'].append(layer_grad)
            samples_grads[i][key] = layer_grad
            i += 1


    return samples_grads


# def generate_private_grad(model,model_tmp, loss_fn,samples,targets,
#                            optimizer,optimizer_tmp,inner_n_epochs,inner_batch_size,sigma,const_C,val_loader,device):
def generate_private_grad(model, loss_fn,samples,targets,
                           optimizer,inner_n_epochs,inner_batch_size,sigma,const_C,val_loader,device):
    '''
        We generate private grad given a batch of samples (samples,targets) in batchclipping mode for classical mini-batch SGD
    '''

    #prepare a new dataloader based on given mini-batch
    mini_dataset = TensorDataset(samples,targets)
    mini_dataloader = DataLoader(mini_dataset,inner_batch_size,shuffle=True)

    #save the starting model state for compute the sum of gradients in final step
    # model_state_start = model.state_dict().
    #save the model config
    model_state_start = model.state_dict()
    optimizer_state = optimizer.state_dict()
    # scheduler_state = scheduler.state_dict()
    dict_state = dict()
    # dict_state["epoch"] = epoch
    # dict_state["sigma"] = sigma
    # dict_state["const_C"] = const_C
    # dict_state["model_state"] = model_state_start
    # dict_state["optimizer_state"] = optimizer_state
    # # ckpt_PATH = "./checkpoint.pt"
    # torch.save(dict_state, ckpt_PATH)
    # checkpoint = torch.load(ckpt_PATH, weights_only=False)
    # model_tmp
    # optimizer_tmp = optim.SGD()
    # model_tmp.load_state_dict(checkpoint['model_state'])
    # model_tmp.to(device)
    # optimizer_tmp = optim.SGD(model_tmp.parameters(),lr=0.001)
    # optimizer_tmp.load_state_dict(checkpoint['optimizer_state'])
    # dict_state["scheduler_state"] = scheduler_state
    #TODO: Clone model and optimizer for inner loop
    # ref:  https://discuss.pytorch.org/t/does-deepcopying-optimizer-of-one-model-works-across-the-model-or-should-i-create-new-optimizer-every-time/14359/7
    model_tmp = copy.deepcopy(model)
    # input(model_tmp)
    optimizer_tmp = type(optimizer)(model_tmp.parameters(), lr=optimizer.defaults['lr'])
    # optimizer_tmp = type(optimizer)(model_tmp.parameters(), lr=0.1)
    optimizer_tmp.load_state_dict(optimizer.state_dict())
    # optimizer_tmp = optim.SGD(model_tmp.parameters(), lr=0.01)
    #training the model with given sub-dataset
    for _ in range(1, inner_n_epochs + 1):
        for inputs,labels in mini_dataloader:
            # compute the gradient of the whole batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_tmp(inputs)
            loss = loss_fn(outputs, labels)
            optimizer_tmp.zero_grad()
            # optimizer.zero_grad()
            loss.backward()
            optimizer_tmp.step()
            # optimizer.step()
        for param_group in optimizer_tmp.param_groups:
            param_group["lr"] = param_group["lr"]*0.9


    #print the test accuracy
    print("INNER LOOP testing ACCURACY")
    print("training size=",len(mini_dataloader.dataset))
    correct = 0
    with torch.no_grad():
        for data in mini_dataloader:
            images, labelsx = data
            images, labelsx = images.to(device), labelsx.to(device)
            outputsx = model_tmp(images)
            _, predicted = torch.max(outputsx, 1)
            c = (predicted == labelsx).squeeze()
            correct += c.sum()
    print("train_acc=",correct / len(mini_dataloader.dataset))
    #   if epoch == 1 or epoch % 1 == 0:
    #       print('Inner Epoch {}, Val accuracy {}'.format(epoch, correct / len(cifar10_val)))

    #extract the sum of gradients, i.e., sum_grads = model.state_dict_last - model.state_dict_start
    # sum_grads contains tensor
    model_state_last = model_tmp.state_dict()

    # sum_grads = OrderedDict()
    # for layer in model_state_start.keys():
    #      sum_grads[layer] = model_state_last[layer] - model_state_start[layer]
    #      sum_grads[layer] = sum_grads[layer].float()
    # Computing aH
    for param1, param2 in zip(model.parameters(), model_tmp.parameters()):
        # param1.grad= torch.sub(param2.data,param1.data).div(args.lr) #aH = (W_m - W_0)/eta
        param1.grad= torch.sub(param2.data,param1.data) #aH = (W_m - W_0)

    #generate private grad per layer
    mean = 0
    std = sigma*const_C
    norm_type = 2.0
    #clipping the gradient
    #https://discuss.pytorch.org/t/how-to-clip-grad-norm-grads-from-torch-autograd-grad/137816/2
    # for layer, grad in sum_grads.items():
    for param in model.parameters():
        grad = param.grad
        #clip the gradients
        max_norm = const_C #clipping constant C
        # print(grad.detach())
        # print(type(grad.detach()))
        # print(norm_type)
        # print(type(norm_type))
        total_norm = torch.norm(grad.detach(), norm_type).to(device)
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        grad.detach().mul_(clip_coef_clamped)
        #generate the noise and add it to the clipped grads
        #generate the noise ~ N(0,(C\sigma)^2I)
        #std -- is C\sigma as explain this in wikipage https://en.wikipedia.org/wiki/Normal_distribution N(mu,\sigma^2) and sigma is std
        noise = torch.normal(mean=mean, std=std, size=grad.shape).to(device)
        #generate private gradient per layer
        #TODO: UNCOMMENT AFTER TEST
        grad = grad + noise
        #####
        param.grad = grad

    #reset the model
    # model.load_state_dict(model_state_start)
    # update the model.param.grad with noisy grads
    # for layer, param in model.named_parameters():
    #     param.grad = sum_grads[layer]
    return 0



# def training_loop(outer_n_epochs, optimizer,optimizer_tmp, model, model_tmp,
#                   loss_fn, inner_n_epochs, inner_batch_size, 
#                   lr_outer, sigma, const_C, train_loader, val_loader,
#                 device):
def training_loop(outer_n_epochs, optimizer, model,
                  loss_fn, inner_n_epochs, inner_batch_size, 
                  lr_outer, sigma, const_C, train_loader, val_loader,
                device):
    train_acc = []
    test_acc = []
    for epoch in range(1, outer_n_epochs + 1):
        loss_train = 0.0

        for images, labels in train_loader:
          images, labels = images.to(device), labels.to(device) 
          outputs = model(images)
          loss = loss_fn(outputs, labels)
          loss_train += loss.item()

          optimizer.zero_grad()
          '''
            generate_private_grad(model,loss_fn,imgs,labels,sigma,const_C)
              1. Compute the grad per sample
              2. Clipping the grad per sample
              3. Aggregate the clipped grads and add noise to sum of clipped grads
              4. Update the model.grad. This helps optimizer.step works as normal.
          '''
        #   loss.backward()
        #   generate_private_grad(model,model_tmp, loss_fn,images,labels,
        #                    optimizer,optimizer_tmp,inner_n_epochs,inner_batch_size,sigma,const_C,val_loader,device)
          generate_private_grad(model, loss_fn,images,labels,
                           optimizer,inner_n_epochs,inner_batch_size,sigma,const_C,val_loader,device)
          #update the model
          optimizer.step()
        #   for param in model.parameters():
        #       param.data = param.data - lr_outer*param.grad
        train_correct = 0
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device) 

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                train_correct += c.sum()
        test_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device) 

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                test_correct += c.sum()
        
        if epoch == 1 or epoch % 1 == 0:
            num_train_samples = len(train_loader.dataset)
            num_test_samples = len(val_loader.dataset)
            print("num_test_samples=",num_test_samples)
            print('Epoch {}, Training loss {}, Train_acc {}, Val_acc {}'.format(
                epoch, 
                loss_train / len(train_loader),
                train_correct / num_train_samples,
                test_correct / num_test_samples
                ))
            train_accuracy_np = (train_correct / num_train_samples).cpu().tolist()
            test_accuracy_np = (test_correct / num_test_samples).cpu().tolist()
            train_acc.append(train_accuracy_np)
            test_acc.append(test_accuracy_np)
            # print(type(train_acc[0]))
            # print(type(test_acc[0]))
            # input()

        # before_lr = optimizer.param_groups[0]["lr"]
        # scheduler.step()
        # after_lr = optimizer.param_groups[0]["lr"]
        # print("Epoch %d: SGD lr %f -> %f" % (epoch, before_lr, after_lr))
    return train_acc, test_acc
