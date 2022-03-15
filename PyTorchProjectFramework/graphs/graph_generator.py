import matplotlib.pyplot  as plt
import numpy

import os
import json
if __name__ == "__main__":
    experiment = "DPSGD"
    # setting_file_name = "settings_main_theorem(test)"
    # settings = ["setting_1","setting_2","setting_3"]
    settings = ["setting_3"]
    for setting in settings:
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
            train_accuracy = data["train_accuracy"]
            test_accuracy = data["test_accuracy"]
            epochs = len(train_accuracy)

        print("Plotting graphs for setting : %s" % setting)
        epoch_index = [i for i in range(1, epochs+1)]
        # T_index = [N_c/s * i for i in range(1, epochs+1)]
        index = epoch_index
        # print(eps_dpsgd)
        # input()
        plt.plot(index, train_accuracy, label="train_accuracy (setting 3)")
        plt.plot(index, test_accuracy, label="test_accuracy (setting 3)")
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


