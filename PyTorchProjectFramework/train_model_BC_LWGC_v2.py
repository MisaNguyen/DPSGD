import pickle

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from collections import OrderedDict
from collections import defaultdict
from torch.func import functional_call, vmap, grad

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


def generate_private_grad(model,loss_fn,samples,targets,sigma,const_C,device):
    '''
        We generate private grad given a batch of samples (samples,targets) in batchclipping mode
        The implementation flow is as follows:
            1. samples x0, x1, ..., x(L-1)
            2. compute avg of gradient g = sum(g0, ..., g(L-1))/L
            3. clipped gradient gc
            4. clipped noisy gradient (gc*L + noise)/L

        Finally, we normalize the private gradient and update the model.grad. This step allows optimizer update the model
    '''

    #copute the gradient of the whole batch
    outputs = model(samples)
    loss = loss_fn(outputs, targets)
    model.zero_grad()
    loss.backward()

    #generate private grad per layer
    mean = 0
    std = sigma*const_C
    batch_size = len(samples)
    norm_type = 2.0
    #clipping the gradient
    #https://discuss.pytorch.org/t/how-to-clip-grad-norm-grads-from-torch-autograd-grad/137816/2
    for param in model.parameters():
        #clip the gradients
        max_norm = const_C #clipping constant C
        param.grad = param.grad * batch_size # New in v2
        grad = param.grad
        total_norm = torch.norm(grad.detach(), norm_type).to(device)
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        param.grad.detach().mul_(clip_coef_clamped)
        #generate the noise and add it to the clipped grads
        # grad = param.grad
        #generate the noise ~ N(0,(C\sigma)^2I)
        #std -- is C\sigma as explain this in wikipage https://en.wikipedia.org/wiki/Normal_distribution N(mu,\sigma^2) and sigma is std
        noise = torch.normal(mean=mean, std=std, size=param.grad.shape).to(device)
        #generate private gradient per layer
        param.grad = (param.grad + noise)/batch_size

    return 0

def training_loop(n_epochs, optimizer, model, loss_fn, scheduler,
                  sigma, const_C, train_loader, val_loader, device):
    train_acc = []
    test_acc = []
    for epoch in range(1, n_epochs + 1):
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
          generate_private_grad(model,loss_fn,images,labels,sigma,const_C,device)
          optimizer.step()
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

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: SGD lr %f -> %f" % (epoch, before_lr, after_lr))
    return train_acc, test_acc
