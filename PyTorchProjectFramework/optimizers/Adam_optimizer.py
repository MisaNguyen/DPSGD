import torch.optim as optim
"""
Optimizer callers
"""
def Adam_optimizer(model_parameters,learning_rate):
    optimizer = optim.Adam(model_parameters, lr=learning_rate)
    return optimizer