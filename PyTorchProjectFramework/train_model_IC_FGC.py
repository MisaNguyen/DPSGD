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
        We generate private grad given a batch of samples (samples,targets) as introduced here https://arxiv.org/pdf/1607.00133.pdf
        The implementation flow is as follows:
            1. sample xi
            2. ===> gradient gi
            3. ===> clipped gradient gci
            4. ===> noisy aggregated (sum gci + noise)
            5. ===> normalized 1/B (sum gci + noise)

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

        After that we compute the clipped gradients gci for each gi. In this case we use the following approach proposed here
            #https://www.tutorialspoint.com/python-pytorch-clamp-method
        To do it, we need to create a new field called whole_grad which containing all gradients of layers for a given sample si
        whole_grad allows us to compute the total_norm of sample si and then we can do the clipping

        After computing all clipped gradients, we need to aggregate all the clipped gradient per layer. This step helps us
        to compute the sum (clipped gradient gi) and then we add noise to each entry in the sum (clipped gradient gi)

        Finally, we normalize the private gradient and update the model.grad. This step allows optimizer update the model
    '''

    samples_grads = compute_gradients(model,loss_fn,samples,targets)

    #compute the size of batch for normalizing the private grad as done next steps
    num_samples = len(samples)

    #clipping the per_sample_grad
    for i in range(num_samples):
        norm_type = 2.0
        max_norm = const_C #This is clipping constant C
        #https://discuss.pytorch.org/t/how-to-clip-grad-norm-grads-from-torch-autograd-grad/137816/2
        total_norm = torch.norm(torch.stack([torch.norm(grad.detach(), norm_type) for grad in samples_grads[i]['whole_grad']]), norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        #https://www.tutorialspoint.com/python-pytorch-clamp-method
        #clamp(tensor,min,max)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        #we delete the field "whole_grad" to make the clipping process easier, i.e., less confusing
        samples_grads[i].pop("whole_grad")
        for layer, grad in samples_grads[i].items():
            grad.detach().mul_(clip_coef_clamped)

    #Aggregate clipped grads
    '''
        aggregated_grad_dict looks like as follows if we have two samples s0 and s1 as described above.
            aggreated_grad_dict[key=weight]= {1, 3}
            aggreated_grad_dict[key=bias]= {2, 4}
        To get it, we have to loop through all samples and for each sample, we loop through each layer (key) to get it grad (value)
    '''

    aggregated_grad_dict = defaultdict(list)

    for sample in samples_grads.values():
        for layer, grad in sample.items():
            aggregated_grad_dict[layer].append(grad)

    #generate private grad per layer
    mean = 0
    std = sigma*const_C
    batch_size = num_samples
    for layer, list_grad in aggregated_grad_dict.items():
        #compute the sum of clipped gradients gi
        
        # print("batch_size=",batch_size)
        # print(torch.stack(list_grad).shape)
        aggregated_grad_dict[layer] = torch.sum(torch.stack(list_grad),dim=0)
        # print(aggregated_grad_dict[layer].shape)
        # input()
        #generate the noise ~ N(0,(C\sigma)^2I)
        #std -- is C\sigma as explain this in wikipage https://en.wikipedia.org/wiki/Normal_distribution N(mu,\sigma^2) and sigma is std
        noise = torch.normal(mean=mean, std=std, size=aggregated_grad_dict[layer].shape).to(device)
        #generate private gradient per layer
        
        aggregated_grad_dict[layer] = (aggregated_grad_dict[layer] + noise)/batch_size

    #update the model's grads
    '''
        Because we do not use loss_fn.backward() function to generate model.grad, model.grad is NONE
        We need to update the model.grad to make sure that optim.step() can operate normally
    '''

    for layer, param in model.named_parameters():
        param.grad =  aggregated_grad_dict[layer]

    # return 0

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


        #save the model config
        # model_state = model.state_dict()
        # optimizer_state = optimizer.state_dict()
        # scheduler_state = scheduler.state_dict()
        # dict_state = dict()
        # dict_state["epoch"] = epoch
        # dict_state["sigma"] = sigma
        # dict_state["const_C"] = const_C
        # dict_state["model_state"] = model_state
        # dict_state["optimizer_state"] = optimizer_state
        # dict_state["scheduler_state"] = scheduler_state
        # dict_state["train_loss"] = loss_train / len(train_loader)
        # dict_state["val_acc"] = correct / len(cifar10_val)

        # try:
        #     geeky_file = open(data_path + "epoch_" + str(epoch), 'wb')
        #     pickle.dump(dict_state, geeky_file)
        #     geeky_file.close()

        # except:
        #     print("Something went wrong")
        #print(f"scheduler state: {scheduler_state}")
    return train_acc, test_acc
