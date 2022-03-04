import matplotlib.pyplot  as plt
import numpy

import os
import json
if __name__ == "__main__":
    experiment = "mainbody"
    setting_file_name = "settings_main_theorem(test)"
    settings = ["setting_1","setting_2","setting_3","setting_4","setting_5","setting_6"]
    for setting in settings:
        graph_path = "./graph/" + experiment + "/" + setting
        data_path  = "./data/" + setting_file_name + "/" + experiment + "/" + setting +".json"
        # Check whether the specified path exists or not
        isExist = os.path.exists(graph_path)

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(graph_path)
            print("The new directory is created: %s" % graph_path)

        with open(data_path, "r") as data_file:
            data = json.load(data_file)
            delta = data["delta"]
            eps_dpsgd = data["eps_dpsgd"]
            eps_fdp = data["eps_fdp"]
            sigma = data["sigma"]
            mu_fdp = data["mu_fdp"]
            N_c = data["num_examples"]
            s = data["sampling_batch"]
            epochs = len(eps_fdp)

        print("Plotting graphs for setting : %s" % setting)
        epoch_index = [i for i in range(1, epochs+1)]
        T_index = [N_c/s * i for i in range(1, epochs+1)]
        index = T_index
        # print(eps_dpsgd)
        # input()
        plt.plot(index, eps_dpsgd, label="eps_dpsgd")
        plt.plot(index, eps_fdp, label="eps_fdp")
        # plt.plot(epoch_index, sigma, label="sigma")
        plt.title('epsilon over T ("delta = %f, s = %f" )' % (delta,s))
        plt.xlabel('T')
        plt.ylabel('epsilon')
        plt.legend()
        plt.savefig(graph_path + "/eps.png")
        # plt.show()
        plt.clf()
        plt.plot(index, sigma, label="sigma")
        plt.title('sigma over T ("delta = %f, s = %f" )' % (delta,s))
        plt.xlabel('T')
        plt.ylabel('sigma')
        plt.legend()
        plt.savefig(graph_path + "/sigma.png")
        # plt.show()

        mult_factor = [eps_dpsgd[i]/eps_fdp[i] for i in range(len(eps_fdp))]
        # print(mult_factor)
        # print(sigma)
        plt.clf()
        min_fact = min(mult_factor)
        min_fact_index = mult_factor.index(min_fact)
        # print(min_fact)
        # print(min_fact_index)
        plt.plot(index, mult_factor, label="mult_factor")
        plt.title('mult_factor over T ("delta = %f, s = %f" )' % (delta,s))
        plt.xlabel('T')
        plt.ylabel('eps_dpsgd/eps_fdp')
        plt.legend()
        plt.savefig(graph_path + "/mult_factor.png")
        # plt.show()
        plt.clf()



