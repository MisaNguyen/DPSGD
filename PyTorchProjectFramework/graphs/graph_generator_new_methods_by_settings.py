import matplotlib
import matplotlib.pyplot  as plt
import numpy

import os
import json

import numpy as np

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 22}
#
# matplotlib.rc('font', **font)
plt.rcParams.update({'font.size': 40})
def get_data(data_path):
    data = {}
    if (os.path.exists(data_path)):
        with open(data_path, "r") as data_file:
            data = json.load(data_file)
            return data
    else:
        print("file not found: ", data_path)
    return None

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

if __name__ == "__main__":
    setting_path = "./new_result"
    # setting_filename = "settings_clipping_exp_cifar10_dpsgd_opacus"
    setting_filename = "settings_convnet"
    setting_name = "setting_0"
    model = "convnet"
    # mode = None
    cmap = get_cmap(30)
    methods = ["IC_FGC", "IC_LWGC", "IC_DLWGC",
               "BC_FGC/v1", "BC_LWGC/v1", "BC_DLWGC/v1"]
    for idx, method in enumerate(methods):
        data_path = setting_path + "/" + model + "/" + method + "/" + setting_filename + ".json"
        print("data_path=",data_path)
        data = get_data(data_path)
        if (data is None):
            continue
        train_accuracy = data[setting_name]["train_accuracy"]
        test_accuracy = data[setting_name]["test_accuracy"]
        
        plt.plot(test_accuracy, label=method, color=cmap(idx*4))
        title = "BS:" + str(data[setting_name]["batch_size"]) \
                + ", Lr:" + str(data[setting_name]["learning_rate"]) \
                + ", Sigma:" + str(data[setting_name]["noise_multiplier"]) \
                + ", C:" + str(data[setting_name]["max_grad_norm"]) 
    plt.title(title)
    plt.legend(loc=2, prop={'size': 12})
    plt.show()