import pickle

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from collections import OrderedDict
from collections import defaultdict
from torch.func import functional_call, vmap, grad


def generate_private_grad(model,loss_fn,samples,targets,sigma,dict_const_Ci,device):
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

    outputs = model(samples)
    loss = loss_fn(outputs, targets)
    model.zero_grad()
    loss.backward()

    #generate private grad per layer
    mean = 0
    batch_size = len(samples)
    norm_type = 2.0

    for layer, param in model.named_parameters():
        max_norm = dict_const_Ci[layer] #This is clipping constant Ci for layer_i
        grad = param.grad
        total_norm = torch.norm(grad, norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        #https://www.tutorialspoint.com/python-pytorch-clamp-method
        #clamp(tensor,min,max)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        param.grad.detach().mul_(clip_coef_clamped)

        #generate the noise and add it to the clipped gradients
        grad = param.grad
        std = sigma*dict_const_Ci[layer]
        #generate the noise ~ N(0,(C\sigma)^2I)
        #std -- is C\sigma as explain this in wikipage https://en.wikipedia.org/wiki/Normal_distribution N(mu,\sigma^2) and sigma is std
        noise = torch.normal(mean=mean, std=std, size=grad.shape).to(device)
        #generate private gradient per layer
        param.grad = (grad*batch_size + noise)/batch_size
    # return 0

def generate_layerwise_clipping_constants(model,optimizer,loss_fn,data_surrogate_loader,const_C,device):
    '''
      We compute the layerwise clipping constant Ci based on data_surrogate
      Step 1. We compute the layer norm Ci
      Step 2. We redefine Ci = Const_C * (Ci/(max_i Ci))
    '''
    for imgs, labels in data_surrogate_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()

    dict_clipping_const = dict()
    norm_type = 2.0
    maxC = 0.0
    for layer, param in model.named_parameters():
        grad = param.grad
        dict_clipping_const[layer] = torch.norm(grad, norm_type)
        if(dict_clipping_const[layer] > maxC):
          maxC = dict_clipping_const[layer]

    #delete the information in the model.param.grad
    optimizer.zero_grad()

    #normalize the clipping constant Ci
    for layer in dict_clipping_const.keys():
        dict_clipping_const[layer] = const_C*(dict_clipping_const[layer]/maxC)

    return dict_clipping_const


def training_loop(n_epochs, optimizer, model, loss_fn, scheduler,
                  sigma, const_C, train_loader, val_loader, data_surrogate_loader, device):
    train_acc = []
    test_acc = []
    for epoch in range(1, n_epochs + 1):
        #generate the layerwise clipping constant Ci
        dict_const_Ci = generate_layerwise_clipping_constants(model,optimizer,loss_fn,data_surrogate_loader,const_C,device)
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
            generate_private_grad(model,loss_fn,images,labels,sigma,dict_const_Ci,device)
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