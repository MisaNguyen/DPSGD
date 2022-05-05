import argparse
import numpy as np
import torch
import torchvision
# import torch.nn.functional as F
import torch.nn as nn
# from torch.optim.lr_scheduler import StepLR
# from utils.progression_bar import progress_bar
# from datasets import create_dataset
# from utils import parse_configuration
# import math
# from models import create_model
# import time

import matplotlib.pyplot as plt

"""Opacus"""
# from opacus.utils.batch_memory_manager import BatchMemoryManager

""" Schedulers """
from scheduler.learning_rate_scheduler import StepLR
from scheduler.gradient_norm_scheduler import StepGN_normal
# from scheduler.noise_multiplier_scheduler import StepLR

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

def train(args, model, device, train_loader, optimizer_name, epoch,visualizer,is_diminishing_gradient_norm, is_individual):
    model.train()
    train_loss = 0
    train_correct = 0
    total = 0
    output = 0
    loss = 0
    # Get optimizer

    if optimizer_name == "DPSGD":
        optimizer = DPSGD_optimizer(model.parameters(),args.lr,
                                                    args.noise_multiplier,args.max_grad_norm,
                                    args.batch_size)
    elif optimizer_name == "SGD":
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        optimizer = SGD_optimizer(model.parameters(),args.lr)
    elif optimizer_name == "Adam":
        optimizer = Adam_optimizer(model.parameters(), args.lr)
    # train_accuracy = np.array()
    # for batch_idx, (data, target) in enumerate(train_loader):
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print(data.shape)
        # print(data.shape[0])
        # print(data.shape[1])
        # input()
        if optimizer_name == "DPSGD":
            """
            Individual-clipping
            """
            if(is_individual == True):
                # optimizer = DPSGD_optimizer(model.parameters(),args.lr,
                #                                             args.noise_multiplier,args.max_grad_norm)
                # Reset the sum_grads
                for param in model.parameters():
                    param.accumulated_grads = None
                # input(len(batch[0]))
                # input(len(batch[1]))

                for sample_idx in range(0,len(data)):
                    sample_x, sample_y = data[sample_idx],target[sample_idx]
                # sample_y = target[sample_idx]

                    # Reset gradients to zero
                    optimizer.zero_grad()
                    # Calculate the loss
                    previous_output = output
                    previous_loss = loss
                    output = model(sample_x[None, ...]) # input as batch size = 1
                    # input(output.shape)
                    # input(target.shape)
                    # input(sample_y)
                    loss = nn.CrossEntropyLoss()(output, sample_y[None, ...])

                    # if np.isnan(loss.cpu().detach().numpy()):
                    #     print("NaN loss")
                    #     print(batch_idx)
                    #     print(sample_x[None, ...])
                    #     print(sample_y[None, ...])
                    #     # imshow(torchvision.utils.make_grid(sample_x.cpu()))
                    #     print(output)
                    #     print("previous loss", previous_loss)
                    #     print("previous output", previous_output)
                    #     input()

                    # Loss back-propagation
                    loss.backward()


                    # train_loss += loss.item()
                    # prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
                    # total += target.size(0)
                    #
                    # # train_correct incremented by one if predicted right
                    # train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
                    #
                    # progress_bar(batch_idx, len(train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                    #              % (train_loss / (batch_idx + 1), 100. * train_correct / total, train_correct, total))
                    # Clip each parameter's per-sample gradient
                    for param in model.parameters():
                        # Clip the sample grad
                        # param.register_hook(lambda grad: torch.clamp(grad, -args.max_grad_norm, args.max_grad_norm))
                        # Detach the sample gradient
                        per_sample_grad = param.grad.detach().clone()
                        # torch.nn.utils.clip_grad_norm_(per_sample_grad, max_norm=args.max_grad_norm)  # in-place
                        # input(param.accumulated_grads)
                        # Add sample's gradient to sum_grad
                        if(param.accumulated_grads == None):
                            param.accumulated_grads = per_sample_grad
                        else:
                            param.accumulated_grads.add_(per_sample_grad)
                        # input(param.accumulated_grads)
                        # input(param.accumulated_grads)
                        # param.accumulated_grads.append(per_sample_grad)
                        # print(torch.stack(param.accumulated_grads, dim=0).shape)
                # Aggregate gradients
                # model.to("cpu")
                # with torch.no_grad():
                for param in model.parameters():
                    param.grad = torch.mul(param.accumulated_grads,1/args.batch_size)
                    # input(len(param.accumulated_grads))
                    # accumulated_grads = torch.stack(param.accumulated_grads, dim=0).sum(dim=0)
                    # input(param.grad.shape)
                    # print(accumulated_grads)
                    # input(param.size())
                    # input(param.grad.size())
                    # input(accumulated_grads.sum(dim=0).size())

                    # param.grad = torch.sum(torch.stack(param.accumulated_grads), dim=0)
                    # param.grad = torch.stack(param.accumulated_grads, dim=0).sum(dim=0)
                """
                Diminishing gradient norm mode.
                """
                # if (mode == True):
                #     new_noise_multiplier =
                #     new_max_grad_norm =
                #     optimizer = DPSGD_optimizer(model.parameters(),args.lr,
                #                                                 args.noise_multiplier,args.max_grad_norm)
                # model.to(device)
            else:
                """
                Batch-clipping
                """
                optimizer.zero_grad()
                # Calculate the loss
                # previous_output = output
                # previous_loss = loss
                output = model(data) # input as batch size = 1
                # input(output.shape)
                # input(target.shape)
                # input(sample_y)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()

                #Batch clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm) # in-place computation
                # torch.nn.utils.clip_grad_value_(model.parameters(), max_norm=args.max_grad_norm) # in-place computation
                # for p in model.parameters():
                #     p.register_hook(lambda grad: torch.clamp(grad, -args.max_grad_norm, args.max_grad_norm))
                # for param in model.parameters():
                #     torch.nn.utils.clip_grad_norm_(param.grad, max_norm=args.max_grad_norm) # in-place computation
        elif (optimizer_name == "SGD" or optimizer_name == "Adam") :

            # optimizer = SGD_optimizer(model.parameters(),args.lr)
            optimizer.zero_grad()
            # for param in model.parameters():
            #     input(param.grad)
            #     break
            # Calculate the loss
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            # loss = F.cross_entropy(output, target)
            # Loss back-propagation = Calculate gradients
            loss.backward()
            # if np.isnan(loss.cpu().detach().numpy()):
            #     print("NaN loss")
            #     print(batch_idx)
            #     print(output)
            #     print(target)
            #     input()
            # for param in model.parameters():
            #     input(param.grad)
            #     break

        # Get scheduler
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        # Calculate gradient step
        optimizer.step()
        # Decrease learning rate using scheduler
        scheduler.step()

        # Trainning Log
        # output = model(data)
        # loss = nn.CrossEntropyLoss()(output, target)
        # train_loss += loss.item()
        # prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
        # total += target.size(0)
        #
        # # train_correct incremented by one if predicted right
        # train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
        #
        # progress_bar(batch_idx, len(train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1), 100. * train_correct / total, train_correct, total))


        # data, target = data.to(device), target.to(device)
        # print("here",data)
        # input(data.shape)
        # input(data.shape)
        # input(target.shape)



        # optimizer.zero_grad()
        # output = model(data)
        # loss = nn.CrossEntropyLoss()(output, target)
        # loss = F.nll_loss(output, target)

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

        if batch_idx % args.log_interval == 0:
            print("Training using %s optimizer" % optimizer_name)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            # Trainning Log
            output = model(data)

            loss = nn.CrossEntropyLoss()(output, target)

            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced

            # print(prediction[1])
            # print(target)
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
            # train_accuracy.append(train_correct / total)
            # losses = model.get_current_losses()
        #     visualizer.print_current_losses(epoch, batch_idx * len(data), len(train_loader.dataset), loss)

        # visualizer.plot_current_losses(epoch, batch_idx / len(train_loader), loss)

        if args.dry_run:
            break
    return train_correct / total

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    train(args.configfile)
