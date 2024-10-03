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
plt.rcParams.update({'font.size': 10})
def get_data(data_path):
    if (os.path.exists(data_path)):
        with open(data_path, "r") as data_file:
            data = json.load(data_file)
            train_accuracy = data["train_accuracy"]
            test_accuracy = data["test_accuracy"]
        return train_accuracy, test_accuracy
    else:
        print("file not found: ", data_path)
    return None,None

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
if __name__ == "__main__":
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # loading SGD data
    settings = [
        {
            # Setting 0
            "settings_path": "utility_graph",
            "C": [0.001 * pow(2,i) for i in range(1,17)],
            "sigma": 0.25,
            "ss": 64
        }
    ]
    # mode = None
    cmap = get_cmap(30)
    # data_folder = "utility_graph"
    data_folder = "utility_graph/opacus" # Opacus
    models = ["convnet","nor_convnet", "group_nor_convnet"]
    # models = ["group_nor_convnet"]
    # models = ["convnet","nor_convnet"]
    sigma_folders = ["sigma_small","sigma_mid","sigma_large"]
    # model_folders = ["nor_convnet","group_nor_convnet","GN_BN_convnet"]
    sigma_folder = sigma_folders[0]
    # model_folder = ""
    sigma = 0.01
    # data_folder = data_folder + "/" + sigma_folder
    # data_folder = data_folder + "/" + model_folder
    file_prefixes = "utility_graph_data"
    pre_train = False
    dataset = "CIFAR10"
    base_acc = "50"
    clipping_modes = ["layerwise","all"]
    lr = 0.001
    ss = 64

    # Get models and settings
    # setting_index = 0# 0,3,6
    # s_index =0
    # models_index = 7
    # models_index = 2
    # models_index = 6
    # settings_path, C_arr, sigma, ss = settings[setting_index]["settings_path"], \
    #                                     settings[setting_index]["C"], \
    #                                     settings[setting_index]["sigma"], \
    #                                     settings[setting_index]["ss"]

    # Partition setting
    partition = False
    graph_path = "./graph/utility_graph"
    # graph_path = "./graph/utility_graph/opacus"

    # plt.title(title)

    fig, axs = plt.subplots(len(models), len(clipping_modes))
    count = 0
    for model_idx, model in enumerate(models):
        for clipping_modes_idx, clipping_mode in enumerate(clipping_modes):
            C_arr = []
            IC_train = []
            IC_test = []
            BC_train = []
            BC_test = []
            ax = axs[model_idx][clipping_modes_idx]
            if pre_train:
                file_path = "./" + data_folder + "/" + model + "/" + file_prefixes + "_" + dataset + "_" \
                            + clipping_mode + "_" + base_acc
            else:
                file_path = "./" + data_folder + "/" + model + "/" + file_prefixes + "_" + dataset + "_" \
                            + clipping_mode + "_no_pre_train"
            if sigma == 0:
                file_path = file_path + "_noNoise"
            file_path = file_path + ".json"
            print(file_path)
            title =  model + " + " + clipping_mode
            # plt.xlabel("C")
            # plt.ylabel("accuracy")
            ax.set_title(title)
            ax.set_xlabel("C")
            ax.set_ylabel("Accuracy")
            if (os.path.exists(file_path)):
                print("opening:",file_path)
                with open(file_path, "r") as data_file:
                    data = json.load(data_file)
                    base_model_test_acc = data[0]["base_model_test_acc"]
                    for i in range(1, len(data)):
                        C_arr.append(data[i]["C"])
                        if (data[i]["IC_train"] != [] and data[i]["IC_test"] != []):
                            IC_train.append(data[i]["IC_train"][-1])
                            IC_test.append(data[i]["IC_test"][-1])
                            cmap_color= cmap(4)
                            print(C_arr)
                            print(IC_train)
                            l1 = ax.plot(C_arr, IC_train, label="IC_train", color=cmap_color)
                            cmap_color= cmap(8)
                            l2 = ax.plot(C_arr, IC_test, label="IC_test", color=cmap_color)

                        if (data[i]["BC_train"] != [] and data[i]["BC_test"] != []):
                            BC_train.append(data[i]["BC_train"][-1])
                            BC_test.append(data[i]["BC_test"][-1])
                            cmap_color= cmap(16)
                            l3 = ax.plot(C_arr, BC_train, label="BC_train", color=cmap_color)
                            cmap_color= cmap(32)
                            l4 = ax.plot(C_arr, BC_test, label="BC_test", color=cmap_color)
    # lines_labels = [ax[0].get_legend_handles_labels() for ax in fig.axes]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # fig.legend(lines, labels)
    handles, labels = axs[0][0].get_legend_handles_labels()
    # fig.legend(handles[:4], labels[:4], loc='upper right')
    fig.legend(handles[:4], labels[:4], loc='upper right')
    # fig.suptitle("Base model accuracy: " + str(base_model_test_acc))
    fig.suptitle(f'Base model accuracy (pretrain: {pre_train}, sigma: {sigma}): ')
    # plt.legend()
    plt.show()
                    # print("C=",C_arr)
                    # input()
                    # train_accuracy = data["train_accuracy"]
                    # test_accuracy = data["test_accuracy"]

