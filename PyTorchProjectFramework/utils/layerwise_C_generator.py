import torch.nn as nn
"""
Compute the master C for each layer given a set of the data at certain epoch
"""

def compute_layerwise_C(C_dataset, model, epochs, device, optimizer, C_start, update_mode=False):
    for epoch in range(epochs):
        data, target = C_dataset
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
        if(update_mode):
            optimizer.step()
    each_layer_C = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # layer_name = "layer_" + str(name)
            current_layer_norm = param.grad.data.norm(2).clone().detach()

            if not each_layer_C:
                each_layer_C.append(C_start)
            else:
                C_ratio = current_layer_norm / prev_layer_norm

                each_layer_C.append(each_layer_C[-1]*float(C_ratio))
            prev_layer_norm = current_layer_norm
    return each_layer_C

