import matplotlib.pyplot  as plt
import numpy

import os
import json


if __name__ == "__main__":
    # loading SGD data
    settings = [
        {
            # Setting 0
            "settings_path": "settings_clipping_exp_cifar10_dpsgd",
            "Cs": [0.1,0.05,0.01,0.005,0.5,1.0],
            "sigma": 2,
            "s_start": 64
        },
        {
            # Setting 1
            "settings_path": "settings_clipping_exp_cifar10_dpsgd_new",
            "Cs": [1.0,1.5,2,2.5,3,3.5],
            "sigma": 2,
            "s_start": 256
        },
        {
            # Setting 2
            "settings_path": "settings_clipping_exp_cifar10_dpsgd_large_C",
            "Cs": [6.0,7.0,8.0,9.0,10.0,20.0],
            "sigma": 2,
            "s_start": 64
        },
        {
            # Setting 3
            "settings_path": "settings_clipping_exp_cifar10_dpsgd_sigma_4",
            "Cs": [0.1,0.05,0.01,0.005,0.5,1.0],
            "sigma": 4,
            "s_start": 64
        },
        {
            # Setting 4
            "settings_path": "settings_clipping_exp_cifar10_dpsgd_new_sigma_4",
            "Cs": [1.0,1.5,2,2.5,3,3.5],
            "sigma": 4,
            "s_start": 256
        },
        {
            # Setting 5
            "settings_path": "settings_clipping_exp_cifar10_dpsgd_large_C_sigma_4",
            "Cs": [6.0,7.0,8.0,9.0,10.0,20.0],
            "sigma": 4,
            "s_start": 64
        },
        {
            # Setting 6
            "settings_path": "settings_clipping_exp_cifar10_dpsgd_sigma_8",
            "Cs": [0.1,0.05,0.01,0.005,0.5,1.0],
            "sigma": 8,
            "s_start": 64
        },
        {
            # Setting 7
            "settings_path": "settings_clipping_exp_cifar10_dpsgd_new_sigma_8",
            "Cs": [1.0,1.5,2,2.5,3,3.5],
            "sigma": 8,
            "s_start": 256
        },
        {
            # Setting 8
            "settings_path": "settings_clipping_exp_cifar10_dpsgd_large_C_sigma_8",
            "Cs": [6.0,7.0,8.0,9.0,10.0,20.0],
            "sigma": 8,
            "s_start": 64
        },
    ]
    data_folder = "data_sum"
    mode = "shuffling"
    # mode = "subsampling"
    # clipping_mode = "all"
    clipping_mode = "layerwise"
    draw_DPSGD_IC_case = False
    draw_SGD_case = False
    draw_DPSGD_BC_case = True
    draw_mixing_case = False
    enable_mu = False
    draw_training_acc = False
    models = ["Lenet", "convnet","nor_convnet","BNF_convnet", "AlexNet",
              "resnet18", "resnet34","resnet50","squarenet"]
    # Get models and settings
    setting_index = 0
    index =1
    models_index = 5
    min_range = 0 # min = 0
    max_range = 6 # max = 6
    model_name = models[models_index]
    settings_path, Cs, sigma, s_start = settings[setting_index]["settings_path"], \
                                        settings[setting_index]["Cs"], \
                                        settings[setting_index]["sigma"], \
                                        settings[setting_index]["s_start"]
    # Partition setting
    partition = False
    s_arr = [s_start * pow(2, i) for i in range(min_range,max_range)]
    s = s_arr[index-1]
    lr = 0.025
    settings = ["setting_" + str(5*i+index) for i in range(min_range,max_range)]
    # settings = ["setting_4", "setting_9", "setting_14","setting_19","setting_24"]
    # settings = ["setting_4", "setting_9", "setting_14","setting_19","setting_24","setting_29"]
    # settings = ["setting_16"]
    if (mode != None):
        base_path = "./" + data_folder + "/" + settings_path + "_" + mode
    else:
        base_path = "./" + data_folder + "/" + settings_path + "/" + model_name
    graph_path = "./graph/" + settings_path + '/C_compare'
    for setting_idx, setting in enumerate(settings):
        C = Cs[setting_idx]
        # experiment = "SGD"


        # Check whether the specified path exists or not
        isExist = os.path.exists(graph_path)

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(graph_path)
            print("The new directory is created: %s" % graph_path)
        if(draw_DPSGD_BC_case):
            experiment = "SGD"
            # bc_data_path  = "./data/" + settings_path + '/' + experiment + '/' + setting +".json"

            bc_data_path  = base_path + '_BC/' + model_name + '/' + experiment \
                            + '/' + clipping_mode + '/BC/' + setting +".json"
            print("bc_data_path:",bc_data_path)
            # print(bc_data_path)
            if (os.path.exists(bc_data_path)):
                with open(bc_data_path, "r") as data_file:
                    data = json.load(data_file)
                    BC_DPSGD_train_accuracy = data["train_accuracy"]
                    BC_DPSGD_test_accuracy = data["test_accuracy"]
                    DPSGD_BC_epochs = len(BC_DPSGD_train_accuracy)
                    DPSGD_BC_epoch_index = [i for i in range(1, DPSGD_BC_epochs+1)]
            else:
                continue

        if(draw_DPSGD_IC_case):
            experiment = "SGD"
            ic_data_path = base_path + '/' + experiment + '/IC/' + setting +".json"
            with open(ic_data_path, "r") as data_file:
                data = json.load(data_file)
                IC_DPSGD_train_accuracy = data["train_accuracy"]
                IC_DPSGD_test_accuracy = data["test_accuracy"]
                DPSGD_IC_epochs = len(IC_DPSGD_train_accuracy)
                DPSGD_IC_epoch_index = [i for i in range(1, DPSGD_IC_epochs+1)]

        plt.subplot(1, 2, 1)
        if(draw_DPSGD_BC_case):
            plt.plot(DPSGD_BC_epoch_index, BC_DPSGD_train_accuracy, label="BC, C= %f" % (C))
            print("BC training acc:", BC_DPSGD_train_accuracy[-1])
        if(draw_DPSGD_IC_case):
            plt.plot(DPSGD_IC_epoch_index, IC_DPSGD_train_accuracy, label="IC, C= %f" % (C))
        plt.title('Train accuracy, lr = %f' % lr)
        plt.legend()

        plt.subplot(1, 2, 2)
        if(draw_DPSGD_BC_case):
            plt.plot(DPSGD_BC_epoch_index, BC_DPSGD_test_accuracy, label="BC,C= %f" % (C))
            print("BC test acc:", BC_DPSGD_test_accuracy[-1])
        if(draw_DPSGD_IC_case):
            plt.plot(DPSGD_IC_epoch_index, IC_DPSGD_test_accuracy, label="IC, C= %f" % (C))


        plt.title('Test accuracy, lr = %f, s = %f, sigma = %f' % (lr,s,sigma))
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        # s = s*2
    if(draw_DPSGD_BC_case and draw_DPSGD_IC_case):
        prefix = "BC_IC"
    else:
        if(draw_DPSGD_BC_case):
            prefix = "BC"
        else:
            prefix = "IC"
    fig_index = setting_index

    file_name =  '/' + model_name + '_' + prefix + '_lr_' + str(lr) + '_s_' + str(int(s)) + '_sigma_' + str(sigma) + '_' + str(fig_index)
    while(os.path.exists(graph_path+ file_name)):
        fig_index = fig_index+1
        file_name =  '/' + model_name + '_' + prefix + '_lr_' + str(lr) + '_s_' + str(int(s)) + '_sigma_' + str(sigma) + '_' + str(fig_index)
    fig = plt.gcf()
    fig.set_size_inches((22, 11), forward=False)
    plt.savefig(graph_path + file_name +".png")
    plt.show()
    # plt.clf()
    # plt.plot(index, sigma, label="sigma")
    # plt.title('sigma over T ("delta = %f, s = %f" )' % (delta,s))
    # plt.xlabel('T')
    # plt.ylabel('sigma')
    # plt.legend()
    # plt.savefig(graph_path + "/sigma.png")
    # plt.show()
    #
    # mult_factor = [eps_dpsgd[i]/eps_fdp[i] for i in range(len(eps_fdp))]
    # # print(mult_factor)
    # # print(sigma)
    # plt.clf()
    # min_fact = min(mult_factor)
    # min_fact_index = mult_factor.index(min_fact)
    # # print(min_fact)
    # # print(min_fact_index)
    # plt.plot(index, mult_factor, label="mult_factor")
    # plt.title('mult_factor over T ("delta = %f, s = %f" )' % (delta,s))
    # plt.xlabel('T')
    # plt.ylabel('eps_dpsgd/eps_fdp')
    # plt.legend()
    # plt.savefig(graph_path + "/mult_factor.png")
    # # plt.show()
    # plt.clf()
    #


