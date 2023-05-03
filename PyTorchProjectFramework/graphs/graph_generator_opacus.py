import matplotlib.pyplot  as plt
import numpy as np
from decimal import Decimal

import os
import json
if __name__ == "__main__":
    # loading SGD data

    # setting_file_name = "settings_main_theorem(test)"
    # settings = ["setting_1","setting_2","setting_3"]
    # settings = ["setting_1","setting_2","setting_3","setting_4"]
    # settings = ["setting_5","setting_6","setting_7","setting_8"]

    # settings = ["setting_2","setting_4","setting_6","setting_8"]
    # sigma = 0.5
    # settings = ["setting_11","setting_21","setting_31","setting_41"]
    # settings = ["setting_12","setting_22","setting_32","setting_42"]
    settings = ["setting_13","setting_23","setting_33","setting_43"]
    # sigma = 1.5
    s_arr = [64,128,256,512]
    lr = 0.025
    C = 1.2
    count = 0
    model = "LeNet"
    experiment = "MNIST"
    graph_path = "./graph/" + experiment
    base_path = "./data_sum/opacus_"+ model +"/"

    fig, ax = plt.subplots(1, 1)
    for setting in settings:
        s = s_arr[count]
        count= count+1
        
        data_path  = base_path + setting +".json"
        # Check whether the specified path exists or not
        isExist = os.path.exists(graph_path)

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(graph_path)
            print("The new directory is created: %s" % graph_path)

        with open(data_path, "r") as data_file:
            data = json.load(data_file)
            eps_delta = data["eps_delta"]
            eps_arr = [eps_delta[i][0] for i in range(len(eps_delta))]

            delta = data["eps_delta"][0][1]
            # eps =data["eps_delta"][0]
            # delta =data["eps_delta"][1]
            sigma_bar = data["sigma_prime"]
            sigma = data["sigma"]
            # print("ABDS",len(eps))

            SGD_test_accuracy = data["test_acc"]
            SGD_epochs = len(SGD_test_accuracy)
            # x = int(len(eps)/SGD_epochs)
            # print("X",x)
            # eps = eps[0::x]

    # loading DPSGD data
    #     print("Testing")
        # setting = "setting_1"
    #     experiment = "DPSGD"
    # # setting_file_name = "settings_main_theorem(test)"
    # # settings = ["setting_1","setting_2","setting_3"]
    #     setting = "setting_14"
    # # for setting in settings:
    #     graph_path = "./graph/" + experiment
    #     data_path  = "./data/" + experiment + '/' + setting +".json"
    #     # Check whether the specified path exists or not
    #     isExist = os.path.exists(graph_path)

        # if not isExist:
        #     # Create a new directory because it does not exist
        #     os.makedirs(graph_path)
        #     print("The new directory is created: %s" % graph_path)
        #
        # with open(data_path, "r") as data_file:
        #     data = json.load(data_file)
        #     DPSGD_train_accuracy = data["train_accuracy"]
        #     DPSGD_test_accuracy = data["test_accuracy"]
        #     DPSGD_epochs = len(DPSGD_train_accuracy)
        #
        # print("Plotting graphs for setting : %s" % setting)
        # DPSGD_epoch_index = [i for i in range(1, DPSGD_epochs+1)]
        SGD_epoch_index = [i for i in range(1, SGD_epochs+1)]
        # T_index = [N_c/s * i for i in range(1, epochs+1)]
        # index = epoch_index
        # print(eps_dpsgd)
        # input()
        # plt.plot(DPSGD_epoch_index, DPSGD_train_accuracy, label="DPSGD_train_accuracy %s" % setting)
        # plt.plot(DPSGD_epoch_index, DPSGD_test_accuracy, label="DPSGD_train_accuracy %s" % setting)


        # plt.subplot(1, 2, 1)
        # print(len(eps))
        # print(SGD_epochs)
        # ax[1].plot(SGD_epoch_index, eps, label="s= %s" % (s))
        # ax[1].xaxis.set_ticks(np.arange(min(SGD_epoch_index), max(SGD_epoch_index)+1,5.0))
        # ax[1].set_xlabel('epoch')
        # ax[1].set_ylabel('eps')
        # ax[1].set_title("Privacy budget, $\delta$ = %s " % (delta))
        # plt.subplot(1, 2, 2)
        ax.plot(SGD_epoch_index, SGD_test_accuracy, label="s= %s" % s)
        ax.xaxis.set_ticks(np.arange(min(SGD_epoch_index), max(SGD_epoch_index)+1, 5.0))
        # ax.set_xticks(1)
        # plt.plot(epoch_index, sigma, label="sigma")
        ax.set_title("Testing accuracy")
        ax.set_xlabel('epoch')
        ax.set_ylabel('accuracy')
        print("s=",s)
        # print("eps_last", eps[-1])
        print("acc_last", SGD_test_accuracy[-1])
    # print(round(eps,2))
    print(delta)
    fig.suptitle("Opacus Performance, lr = %s, C = %s, $\\bar{\sigma}$ = %s, ($\epsilon,\delta$) =(%s,%.2e)"
                 % (lr,C,round(sigma_bar,4),round(eps_arr[-1],2),Decimal(delta)))
    ax.legend()

    fig_name = graph_path + '/' + model +"_" + str(sigma)+  ".png"
    print("saving data to:", fig_name)
    plt.savefig(fig_name)
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


