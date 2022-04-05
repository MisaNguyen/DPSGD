import torch.optim as optim
"""
Optimizer callers
"""
# Note: Should choose small learning_rate
def SGD_optimizer(model_parameters,learning_rate):
    optimizer = optim.SGD(model_parameters, lr=learning_rate)
    return optimizer