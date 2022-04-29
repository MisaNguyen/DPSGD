import matplotlib.pyplot  as plt
import numpy

import os
import json
if __name__ == "__main__":
    # loading SGD data
    settings_path = "settings_lr_exp_cifar10_dpsgd"
    # setting_file_name = "settings_main_theorem(test)"
    # settings = ["setting_" + str(i) for i in range(1,6)]
    # settings = ["setting_" + str(i) for i in range(6,11)]
    # settings = ["setting_" + str(i) for i in range(11,16)]
    # settings = ["setting_" + str(i) for i in range(16,21)]
    lr = 0.1
    s = 512
    draw_IC_case = False
    settings = ["setting_0_c1_s2","setting_0_noclip"]
    # settings = ["setting_1","setting_2","setting_3","setting_4"]
    # settings = ["setting_5","setting_6","setting_7","setting_8"]
    # settings = ["setting_16"]
    for setting in settings:
        experiment = "DPSGD"
        graph_path = "./graph/" + settings_path + '/' + experiment
        data_path  = "./data/" + settings_path + '/' + experiment + '/' + setting +".json"

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
        setting = "setting_0_c01_s2"
        ic_data_path = "./data/" + settings_path + '/' + experiment + '/IC/' + setting +".json"
        with open(ic_data_path, "r") as data_file:
            data = json.load(data_file)
            IC_DPSGD_train_accuracy = data["train_accuracy"]
            IC_DPSGD_test_accuracy = data["test_accuracy"]
            IC_DPSGD_epochs = len(IC_DPSGD_train_accuracy)
        DPSGD_epoch_index = [i for i in range(1, DPSGD_epochs+1)]

        plt.subplot(1, 2, 1)
        plt.plot(DPSGD_epoch_index, DPSGD_train_accuracy, label="BC, s= %f" % (s))
        if(draw_IC_case):
            plt.plot(DPSGD_epoch_index, IC_DPSGD_train_accuracy, label="IC, s= %f" % (s))
        plt.title('Train accuracy, lr = %f' % lr)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(DPSGD_epoch_index, DPSGD_test_accuracy, label="BC,s= %f" % (s))
        if(draw_IC_case):
            plt.plot(DPSGD_epoch_index, IC_DPSGD_test_accuracy, label="IC, s= %f" % (s))

        plt.title('Test accuracy, lr = %f' % lr)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        s = s*2
    plt.savefig(graph_path + '/dpsgd_lr_' + str(lr) +".png")
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


