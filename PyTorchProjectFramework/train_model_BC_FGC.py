import pickle

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from collections import OrderedDict
from collections import defaultdict
from torch.func import functional_call, vmap, grad

def generate_private_grad(model,loss_fn,samples,targets,sigma,const_C,device):
    '''
        We generate private grad given a batch of samples (samples,targets) in batchclipping mode
        The implementation flow is as follows:
            1. samples x0, x1, ..., x(L-1)
            2. compute avg of gradient g = sum(g0, ..., g(L-1))/L
            3. clipped gradient gc =  clip_C (g*L)
            4. clipped noisy gradient (gc + noise)/L
        Finally, we normalize the private gradient and update the model.grad. This step allows optimizer update the model
    '''

    #copute the gradient of the whole batch
    outputs = model(samples)
    loss = loss_fn(outputs, targets)
    model.zero_grad()
    loss.backward() # default = avg; => sum

    #get the list of gradients of the whole batch for computing the total norm
    grads = list()
    for param in model.parameters():
        grads.append(param.grad)

    device = grads[0].device
    norm_type = 2.0
    max_norm = const_C #clipping constant C
    total_norm = torch.norm(torch.stack([torch.norm(grad.detach(), norm_type).to(device) for grad in grads]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    #clipping the gradient
    #https://discuss.pytorch.org/t/how-to-clip-grad-norm-grads-from-torch-autograd-grad/137816/2
    for param in model.parameters():
        param.grad.detach().mul_(clip_coef_clamped)

    #generate private grad per layer
    mean = 0
    std = sigma*const_C
    batch_size = len(samples)
    for param in model.parameters():
        grad = param.grad
        #generate the noise ~ N(0,(C\sigma)^2I)
        #std -- is C\sigma as explain this in wikipage https://en.wikipedia.org/wiki/Normal_distribution N(mu,\sigma^2) and sigma is std
        noise = torch.normal(mean=mean, std=std, size=grad.shape).to(device)
        #generate private gradient per layer
        param.grad = (grad*batch_size + noise)/batch_size
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

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: SGD lr %f -> %f" % (epoch, before_lr, after_lr))
    return train_acc, test_acc