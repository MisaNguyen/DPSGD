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
    setting_indexes = [i for i in range(0,30)]
    
    # mode = None
    cmap = get_cmap(30)
    methods = ["IC_FGC", "IC_LWGC", "IC_DLWGC",
               "BC_FGC/v1", "BC_LWGC/v1", "BC_DLWGC/v1"]
    markers = ["o","v","^","<",">","s"]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for idx, method in enumerate(methods):
        Cs = []
        sigmas = []
        Acc = []
        marker = markers[idx]
        data_path = setting_path + "/" + method + "/" + setting_filename + ".json"
        print("data_path=",data_path)
        data = get_data(data_path)
        if (data is None):
            continue
        for setting_index in setting_indexes:
            setting_name = "setting_" + str(setting_index)
                    
            # train_accuracy = data[setting_name]["train_accuracy"]
            Acc.append(max(data[setting_name]["test_accuracy"]))
            Cs.append(data[setting_name]["max_grad_norm"])
            sigmas.append(data[setting_name]["noise_multiplier"])

            print(str(data[setting_name]["noise_multiplier"]))
            title = "BS:" + str(data[setting_name]["batch_size"]) \
                    + ", Lr:" + str(data[setting_name]["learning_rate"])  
        ax.scatter(Cs, sigmas, Acc,label = method, marker=marker)
        # plt.plot(Cs, Acc, label=method, color=cmap(idx*4))
    ax.set_xlabel('C')
    ax.set_ylabel('Sigma')
    ax.set_zlabel('Test Acc')
    plt.title(title)
    plt.legend(loc=2, prop={'size': 12})
    plt.show()