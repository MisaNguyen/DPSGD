import matplotlib.pyplot  as plt
import numpy

import os
import json
if __name__ == "__main__":
    # loading SGD data

    # setting_file_name = "settings_main_theorem(test)"
    # settings = ["setting_1","setting_2","setting_3"]
    # settings = ["setting_1","setting_2","setting_3","setting_4"]
    # settings = ["setting_5","setting_6","setting_7","setting_8"]
    settings = ["setting_16"]
    for setting in settings:
        experiment = "SGD"
        graph_path = "./graph/" + experiment
        data_path  = "./data/" + experiment + '/' + setting +".json"
        # Check whether the specified path exists or not
        isExist = os.path.exists(graph_path)

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(graph_path)
            print("The new directory is created: %s" % graph_path)

        with open(data_path, "r") as data_file:
            data = json.load(data_file)
            SGD_train_accuracy = data["train_accuracy"]
            SGD_test_accuracy = data["test_accuracy"]
            SGD_epochs = len(SGD_train_accuracy)

    # loading DPSGD data
    #     print("Testing")
        # setting = "setting_1"
        experiment = "DPSGD"
    # setting_file_name = "settings_main_theorem(test)"
    # settings = ["setting_1","setting_2","setting_3"]
        setting = "setting_14"
    # for setting in settings:
        graph_path = "./graph/" + experiment
        data_path  = "./data/" + experiment + '/' + setting +".json"
        # Check whether the specified path exists or not
        isExist = os.path.exists(graph_path)

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(graph_path)
            print("The new directory is created: %s" % graph_path)

        with open(data_path, "r") as data_file:
            data = json.load(data_file)
            DPSGD_train_accuracy = data["train_accuracy"]
            DPSGD_test_accuracy = data["test_accuracy"]
            DPSGD_epochs = len(DPSGD_train_accuracy)

        print("Plotting graphs for setting : %s" % setting)
        DPSGD_epoch_index = [i for i in range(1, DPSGD_epochs+1)]
        SGD_epoch_index = [i for i in range(1, SGD_epochs+1)]
        # T_index = [N_c/s * i for i in range(1, epochs+1)]
        # index = epoch_index
        # print(eps_dpsgd)
        # input()
        plt.plot(DPSGD_epoch_index, DPSGD_train_accuracy, label="DPSGD_train_accuracy %s" % setting)
        plt.plot(DPSGD_epoch_index, DPSGD_test_accuracy, label="DPSGD_train_accuracy %s" % setting)
        plt.plot(SGD_epoch_index, SGD_train_accuracy, label="SGD_train_accuracy %s" % setting)
        plt.plot(SGD_epoch_index, SGD_test_accuracy, label="SGD_train_accuracy %s" % setting)
        # plt.plot(epoch_index, sigma, label="sigma")
        plt.title('Train and test accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(graph_path + '/' + setting +".png")
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


