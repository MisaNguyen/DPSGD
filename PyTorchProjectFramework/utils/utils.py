import torch
import json
import os
import torch.nn as nn
import numpy as np
# ref: https://github.com/pytorch/opacus/blob/5c83d59fc169e93667946204f7a6859827a38ace/opacus/optimizers/optimizer.py#L87
def _generate_noise(
        std: float,
        reference: torch.Tensor,
        generator=None,
        secure_mode: bool = False,
) -> torch.Tensor:
    """
    Generates noise according to a Gaussian distribution with mean 0
    Args:
        std: Standard deviation of the noise
        reference: The reference Tensor to get the appropriate shape and device
            for generating the noise
        generator: The PyTorch noise generator
        secure_mode: boolean showing if "secure" noise need to be generate
            (see the notes)
    Notes:
        If `secure_mode` is enabled, the generated noise is also secure
        against the floating point representation attacks, such as the ones
        in https://arxiv.org/abs/2107.10138 and https://arxiv.org/abs/2112.05307.
        The attack for Opacus first appeared in https://arxiv.org/abs/2112.05307.
        The implemented fix is based on https://arxiv.org/abs/2107.10138 and is
        achieved through calling the Gaussian noise function 2*n times, when n=2
        (see section 5.1 in https://arxiv.org/abs/2107.10138).
        Reason for choosing n=2: n can be any number > 1. The bigger, the more
        computation needs to be done (`2n` Gaussian samples will be generated).
        The reason we chose `n=2` is that, `n=1` could be easy to break and `n>2`
        is not really necessary. The complexity of the attack is `2^p(2n-1)`.
        In PyTorch, `p=53` and so complexity is `2^53(2n-1)`. With `n=1`, we get
        `2^53` (easy to break) but with `n=2`, we get `2^159`, which is hard
        enough for an attacker to break.
    """
    zeros = torch.zeros(reference.shape, device=reference.device)
    if std == 0:
        return zeros
    # TODO: handle device transfers: generator and reference tensor
    # could be on different devices
    if secure_mode:
        torch.normal(
            mean=0,
            std=std,
            size=(1, 1),
            device=reference.device,
            generator=generator,
        )  # generate, but throw away first generated Gaussian sample
        sum = zeros
        for _ in range(4):
            sum += torch.normal(
                mean=0,
                std=std,
                size=reference.shape,
                device=reference.device,
                generator=generator,
            )
        return sum / 2
    else:
        return torch.normal(
            mean=0,
            std=std,
            size=reference.shape,
            device=reference.device,
            generator=generator,
        )


def generate_json_data_for_graph(out_file_path: str,setting : str,train_accuracy : list, test_accuracy : list):
    json_output = {
        "setting": setting,
        "train_accuracy": train_accuracy,
        "test_accuracy" : test_accuracy
    }
    isExist = os.path.exists(out_file_path)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(out_file_path)
        print("The new directory is created: %s" % out_file_path)

    with open(out_file_path + '/' + setting + '.json', "w") as data_file:
        json.dump(json_output, data_file,indent=2)

def json_to_file(out_file_path: str,setting : str, json_str:str):
    isExist = os.path.exists(out_file_path)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(out_file_path)
        print("The new directory is created: %s" % out_file_path)

    with open(out_file_path + '/' + setting + '.json', "w") as data_file:
        json.dump(json_str, data_file,indent=2)

def compute_layerwise_C(C_dataset_loader, model, epochs, device, optimizer, C_start, update_mode=False):
    print("Generating layerwise C values")
    for epoch in range(epochs):
        # print("epoch:",epoch)
        for sample_idx, (data,target) in enumerate(C_dataset_loader):
            # print("sample_idx",sample_idx)
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)

            # compute output
            output = model(data)
            # print(output)
            # compute accuracy
            # preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            # labels = target.detach().cpu().numpy()


            # compute loss
            # previous_loss = loss

            loss = nn.CrossEntropyLoss()(output, target)

            # compute gradient and do SGD step
            loss.backward()
            if (update_mode):
                optimizer.step()
    each_layer_norm = []
    max_norm = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # layer_name = "layer_" + str(name)
            # print("grad", param.grad)
            current_layer_norm = param.grad.data.norm(2).clone().detach()

            # if not each_layer_C:
            #     each_layer_C.append(C_start * current_layer_norm)
            #     max_norm = current_layer_norm
            # else:
            # C_ratio = current_layer_norm / prev_layer_norm
            max_norm = max(max_norm,current_layer_norm)
            each_layer_norm.append(current_layer_norm) # ||a_{h,i}||
            # prev_layer_norm = current_layer_norm
    # input(C_start)
    # input(max_norm)
    # input(each_layer_norm)
    each_layer_C = [C_start*(item/max_norm).cpu().numpy() for item in each_layer_norm]
    return each_layer_C

def compute_layerwise_C_average_norm(C_dataset_loader, model, epochs, device, optimizer, C_start, update_mode=False):
    print("Generating layerwise C values (average)")
    for epoch in range(epochs):
        # print("epoch:",epoch)
        for sample_idx, (data,target) in enumerate(C_dataset_loader):
            # print("sample_idx",sample_idx)
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)

            # compute output
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)

            # compute gradient and do SGD step
            loss.backward()
            if (update_mode):
                optimizer.step()
    each_layer_norm = []
    # max_norm = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # layer_name = "layer_" + str(name)
            current_layer_norm = param.grad.data.norm(2).clone().detach()

            # if not each_layer_C:
            #     each_layer_C.append(C_start * current_layer_norm)
            #     max_norm = current_layer_norm
            # else:
            # C_ratio = current_layer_norm / prev_layer_norm
            # max_norm = max(max_norm,current_layer_norm)
            each_layer_norm.append(np.average(current_layer_norm.cpu().numpy())) # ||a_{h,i}||
            # prev_layer_norm = current_layer_norm
    # input(C_start)
    # input(max_norm)
    each_layer_C = each_layer_norm
    return each_layer_C
