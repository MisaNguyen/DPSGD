import torch.optim as optim
"""
Optimizer callers
"""
def Adadelta_optimizer(model_parameters,learning_rate):
    optimizer = optim.Adadelta(model_parameters, lr=learning_rate)
    return optimizer