import argparse
import copy
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.utils.data import TensorDataset

""" Schedulers """
# from scheduler.learning_rate_scheduler import StepLR
# from scheduler.gradient_norm_scheduler import StepGN_normal
# from scheduler.noise_multiplier_scheduler import StepLR
# from torch.optim.lr_scheduler import ReduceLROnPlateau
""" Optimizers """
from optimizers import *
""" Utils"""
from utils.utils import compute_layerwise_C

def Lr_generator(decay,lr,epoch):
    """
    Create learning_rate sequence generator
    
    Input params:
        decay: learning rate decay
        lr: base learning rate
        epoch: number of training epoch
    Output: Learning rate generator
    """
    lr_sequence = range(epoch)
    for index in lr_sequence:
        yield lr*pow(decay,index)


def sample_rate_generator(multi,lr,epoch):
    """
    Create sample_rate sequence generator
        
    Input params:
        multi: sample rate multiplier
        sample_rate: base sample rate
        epoch: number of training epoch
    Output: sample rate generator
    """
    sr_sequence = range(epoch)
    for index in sr_sequence:
        yield lr*pow(multi,index)

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
    """
    Shows an image (tensor) using matplotlib
    
    Input:
        img: A tensor of shape (3,H,W) to be displayed
    """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
"""
--------------------------------
OPACUS code
--------------------------------
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
    """
    Calculate the accuracy of predictions against labels.

    Parameters:
        preds (torch.Tensor): tensor of predictions
        labels (torch.Tensor): tensor of labels

    Returns:
        float: accuracy of predictions against labels
    """
    return (preds == labels).mean()

def calculate_full_gradient_norm(model):
    """
    Calculate the full gradient norm of the model by flattening the gradient of each layer
    and taking the Euclidean norm of the resulting tensor.
    The gradient of each layer is either a sample gradient (if the layer is a DPM module) or
    a batch gradient (if the layer is a regular PyTorch module).

    Steps:
    1) sqrt(a^2+b^2) = A sqrt(c^2+d^2) = B, sqrt( a^2+b^2 + c^2+d^2) = sqrt(A^2 + B^2)
    2) sqrt(a^2+b^2) = X > C
    3) sqrt(a^2+b^2)/X*C = C
    4) sqrt(a^2 C^2/X^2 + b^2 C^2/X^2) = C
    Returns:
        The full gradient norm of the model
    """
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
    flat_grad_norm = np.sqrt(flat_grad_norm.cpu())
    return flat_grad_norm
"""
END OPACUS code
"""

"""
New method idea: Taking advantage of ALC method without extra privacy cost \sqrt(L)
1) Compute gradient at i-th layer [g_i]_{m_i} = g_i * m_i
2) Clip full gradient: [g]_C = [[g_1]_{m_1},...,[g_L]_{m_L}]_C
3) Compute update: U = [g]_C + N(0,4C^2\sigma^2)
4) Discount update: U_bar = \eta [U_1/m_1,...U_L/m_L] where U = [U_1,U_2,...,U_L]
"""
def DP_train_weighted_FGC(args, model, device, train_loader,optimizer):
    model.train()
    print("Training using %s optimizer" % optimizer.__class__.__name__)
    print("Training with method: weighted full gradient descent")
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

        # print("Minibatch shape", batch_data.shape)
        """ Original SGD updates"""
        for sample_idx, (data,target) in enumerate(micro_train_loader):
            # print("microbatch shape", data.shape)
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

            """
            Add multiplier to gradients (new)
            c_i = C e_i/M
            m_i = C/c_i = M/e_i
            """
            grad_multi = [args.max_grad_norm/c_i for c_i in args.each_layer_C]
            for layer_idx, (name,param) in enumerate(model_clone.named_parameters()):
                param.grad = torch.mul(param.grad,grad_multi[layer_idx])

            """
            Clip entire gradients with args.max_grad_norm
            """
            torch.nn.utils.clip_grad_norm_(optimizer_clone.param_groups[0]['params'],args.max_grad_norm)
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
        """
        Discount updates: U = \sum_{i=1}{m} [U_{i,j}/m_i]_{j=1}^{L}. Note that m_i is the multiply factor
        """
        for layer_idx, (name,param) in enumerate(model_clone.named_parameters()):
            param.grad = torch.div(param.sum_grad,grad_multi[layer_idx])
        """
        Copy sum of clipped grad to the model gradient
        """

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
        if(args.noise_multiplier > 0):
            for layer_idx, (name,param) in enumerate(model.named_parameters()):
                """
                Add Gaussian noise to gradients
                """
                dist = torch.distributions.normal.Normal(torch.tensor(0.0),
                                                         torch.tensor((2 * args.max_grad_norm * args.noise_multiplier)))
                noise = dist.rsample(param.grad.shape).to(device=device)

                # Compute noisy grad
                param.grad = (param.grad + noise).div(len(micro_train_loader))

        """
        Update model with noisy grad
        """
        optimizer.step()

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


def DP_train_classical(args, model, device, train_loader,optimizer):
    """
    Train a model using the classical DP-SGD algorithm.

    Parameters:
        args (argparse.Namespace): command line arguments.
        model (nn.Module): the model to be trained.
        device (torch.device): the device to use for training.
        train_loader (DataLoader): the training data loader.
        optimizer (Optimizer): the optimizer to use.

    Returns:
        The mean accuracy of the model on the training data.
    """
    model.train()
    print("Training using %s optimizer" % optimizer.__class__.__name__)
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
        mini_epochs = 10
        top1_acc_clone=[]
        for mini_epoch in range(mini_epochs):
            print("mini_epoch:", mini_epoch)
            micro_train_loader = torch.utils.data.DataLoader(batch, batch_size=args.microbatch_size,
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
                # print("loss=",loss)
                # compute gradient
                loss.backward()
                # Gradient Descent step
                optimizer_clone.step()

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                acc1_clone = accuracy(preds, labels)
                top1_acc_clone.append(acc1_clone)
                if sample_idx == mini_epochs -1 :

                    # train_loss += loss.item()
                    # prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
                    #
                    # total += batch_target.size(0)
                    #
                    # train_correct += np.sum(prediction[1].cpu().numpy() == batch_target.cpu().numpy())
                    print(
                        f"Model Clone: "
                        f"Loss: {np.mean(losses):.6f} "
                        f"Acc@1: {np.mean(top1_acc_clone):.6f} "
                        f"---------------------------------"
                    )
                #####
        # Computing aH
        for param1, param2 in zip(model.parameters(), model_clone.parameters()):
            # param1.grad= torch.sub(param2.data,param1.data).div(args.lr) #aH = (W_m - W_0)/eta
            param1.grad= torch.sub(param2.data,param1.data) #aH = (W_m - W_0)
            # if (batch_idx == len(train_loader) -1):
            #     print("model clone grad norm!")
            #     # print(param2.grad)
            #     print(calculate_full_gradient_norm(model_clone))
            #     print("model grad norm!")
            #     # print(param1.grad)
            #     print(calculate_full_gradient_norm(model))
            """
            Batch clipping each "batch"
            """
        # if(args.clipping == "layerwise"):
        #     """
        #     Clip each layer gradients with args.max_grad_norm
        #     """
        #     for layer_idx, param in enumerate(model.parameters()):
        #         # print("Before clipping, grad_norm =", param.grad.data.norm(2))
        #         torch.nn.utils.clip_grad_norm_(param, max_norm=args.each_layer_C[layer_idx]) # in-place computation, layerwise clipping
        #
        # elif (args.clipping == "all"):
        #     """
        #     Clip entire gradients with args.max_grad_norm
        #     """
        #     """
        #     Compute flat list of gradient tensors and its norm
        #     """
        #     # flat_grad_norm = calculate_full_gradient_norm(model)
        #     """
        #     Clip all gradients
        #     """
        #     torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'],args.max_grad_norm)
        #     # if (flat_grad_norm > args.max_grad_norm):
        #     #     for param in model.parameters():
        #     #         param.grad = param.grad / flat_grad_norm * args.max_grad_norm
        # else:
        #     raise ValueError("Invalid clipping mode, available options: all, layerwise")
        #
        # # Update model
        # for layer_idx, param in enumerate(model.parameters()):
        #
        #     """
        #     Add Gaussian noise to gradients
        #     """
        #     """--------------STATIC NOISE-----------------"""
        #     # dist = torch.distributions.normal.Normal(torch.tensor(0.0),
        #     #                                          torch.tensor((2 * args.noise_multiplier *  args.max_grad_norm)))
        #     """--------------LAYERWISE NOISE-----------------"""
        #     if(args.clipping=="layerwise"):
        #         dist = torch.distributions.normal.Normal(torch.tensor(0.0),
        #                                                  torch.tensor((2 * args.each_layer_C[layer_idx] *  args.noise_multiplier)))
        #     elif(args.clipping=="all"):
        #         dist = torch.distributions.normal.Normal(torch.tensor(0.0),
        #                                                  torch.tensor((2 * args.max_grad_norm *  args.noise_multiplier)))
        #     # print(param.grad.shape)
        #     noise = dist.rsample(param.grad.shape).to(device=device)
        #
        #     # Compute noisy grad
        #     param.grad = (param.grad + noise).div(len(micro_train_loader)) # len(micro_train_loader) = number of microbatches
        #     # param.grad = param.grad + noise.div(len(micro_train_loader))

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
                # f"Loss: {np.mean(losses):.6f} "
                f"Acc@1: {np.mean(top1_acc):.6f} "
            )
        if args.dry_run:
            break
    return np.mean(top1_acc)

def DP_train(args, model, device, train_loader,optimizer):
    """
    Train a model using the DP-SGD algorithm.

    Args:
        model: The model to be trained.
        device: The device to use for training.
        train_loader: The data loader for the training data.
        optimizer: The optimizer to use.

    Returns:
        The mean accuracy of the model on the training data.
    """
    model.train()
    print("Training using %s optimizer" % optimizer.__class__.__name__)
    loss = 0
    # Get optimizer

    iteration = 0
    losses = []
    top1_acc = []

    for batch_idx, (batch_data,batch_target) in enumerate(train_loader): # Batch loop
        optimizer.zero_grad()
        # copy current model
        model_clone = copy.deepcopy(model)
        # TODO: Remove clone models
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
        micro_train_loader = torch.utils.data.DataLoader(batch, batch_size=args.microbatch_size,
                                                         shuffle=True) # Load each data
        """ Original SGD updates"""
        for sample_idx, (data,target) in enumerate(micro_train_loader):
            # print("microbatch shape", data.shape)
            optimizer_clone.zero_grad()
            iteration += 1
            data, target = data.to(device), target.to(device)
            # compute output
            output = model_clone(data)
            # compute loss
            loss = nn.CrossEntropyLoss()(output, target)
            loss = torch.mul(loss,args.loss_multi)# Adjust losses
            losses.append(loss.item())
            # compute gradient
            loss.backward()

            # Add grad to sum of grad
            """
            Batch clipping each "microbatch"
            """
            print("Clipping method:", args.clipping)
            if(args.clipping == "layerwise"):
                """------------------------------------------------"""
                for layer_idx, param in enumerate(model_clone.parameters()):
                    """
                    Clip each layer gradients with args.max_grad_norm
                    """
                    torch.nn.utils.clip_grad_norm_(param, max_norm=args.each_layer_C[layer_idx])

                    """ 
                    Accumulate gradients
                    """
                    if not hasattr(param, "sum_grad"):
                        param.sum_grad = param.grad

                    else:
                        param.sum_grad = param.sum_grad.add(param.grad)


            elif (args.clipping == "all"):
                """
                Compute flat list of gradient tensors and its norm 
                """
                # flat_grad_norm = calculate_full_gradient_norm(model_clone)
                # print("Current norm = ", flat_grad_norm)
                """
                Clip all gradients
                """
                torch.nn.utils.clip_grad_norm_(optimizer_clone.param_groups[0]['params'],args.max_grad_norm)

                """
                Accumulate gradients
                """
                for param in model_clone.parameters():
                    if not hasattr(param, "sum_grad"):
                        param.sum_grad = param.grad
                    else:
                        param.sum_grad = param.sum_grad.add(param.grad)
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
        if(args.noise_multiplier > 0):
            for layer_idx, (name,param) in enumerate(model.named_parameters()):
                """
                Add Gaussian noise to gradients
                """
                """--------------STATIC NOISE-----------------"""
                # dist = torch.distributions.normal.Normal(torch.tensor(0.0),
                #                                          torch.tensor((2 * args.noise_multiplier *  args.max_grad_norm)))
                """--------------LAYERWISE NOISE-----------------"""

                if(args.clipping=="layerwise"):
                    # if (args.each_layer_C[layer_idx] *  args.noise_multiplier == 0):
                    # print("args.each_layer_C[layer_idx]", args.each_layer_C[layer_idx])
                    # print("args.noise_multiplier", args.noise_multiplier)
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
    """
    Train a model for one epoch. (No clipping, no noise multiplier)

    Parameters:
        args (argparse.Namespace): command line arguments.
        model (nn.Module): the model to train.
        device (torch.device): the device to train on.
        train_loader (DataLoader): the training data loader.
        optimizer (Optimizer): the optimizer to use.
        epoch (int): the current epoch number.

    Returns:
        tuple: a tuple of the mean accuracy and a dictionary of gradient statistics.
    """
    model.train()
    print("Training using %s optimizer" % optimizer.__class__.__name__)

    gradient_stats = {"epoch" : epoch}

    iteration = 0
    losses = []
    top1_acc = []
    loss = None
    for batch_idx, (data,target) in enumerate(train_loader):
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
