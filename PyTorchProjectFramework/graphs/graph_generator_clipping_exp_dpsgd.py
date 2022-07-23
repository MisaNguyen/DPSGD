import matplotlib.pyplot  as plt
import numpy

import os
import json
if __name__ == "__main__":
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # loading SGD data
    settings_path = "settings_clipping_exp_cifar10_dpsgd" # old
    # settings_path = "settings_clipping_exp_cifar10_dpsgd_new"
    # settings_path = "settings_clipping_exp_cifar10_dpsgd_large_C_sigma_4"
    model_name = "convnet"
    # setting_file_name = "settings_main_theorem(test)"
    # settings = ["setting_" + str(i) for i in range(1,6)]
    # settings = ["setting_" + str(i) for i in range(6,11)]
    # settings = ["setting_" + str(i) for i in range(11,16)]
    # settings = ["setting_" + str(i) for i in range(16,21)]
    # settings = ["setting_" + str(i) for i in range(21,26)]
    # settings = ["setting_" + str(i) for i in range(26,31)]
    index =4
    s_index = 1
    # settings = ["setting_" + str(i) for i in range(26,29)]
    # settings.append("setting_30")
    settings = ["setting_" + str(5*index+i) for i in range(s_index,6)]
    # settings = ["setting_0" ]
    lr = 0.1
    Cs = [0.1,0.05,0.01,0.005,0.5,1.0] #old
    # Cs = [1.0,1.5,2,2.5,3,3.5]
    # Cs = [6.0,7.0,8.0,9.0,10.0,20.0]
    C = Cs[index]
    sigma = 4
    s = 32 * pow(2, s_index-1)
    draw_DPSGD_IC_case = True
    draw_SGD_case = False
    draw_DPSGD_BC_case = False
    # settings = ["setting_0_c1_s2","setting_0_noclip"]
    # settings = ["setting_1","setting_2","setting_3","setting_4"]
    # settings = ["setting_1","setting_2"]
    # settings = ["setting_5","setting_6","setting_7","setting_8"]
    # settings = ["setting_16"]
    graph_path = "./graph/" + settings_path + '/clipping'


    # Check whether the specified path exists or not
    isExist = os.path.exists(graph_path)
    base_path = "./data/" + settings_path + "/" + model_name
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(graph_path)
        print("The new directory is created: %s" % graph_path)
    for setting in settings:
        # if (setting == "setting_30"):
        #     s=512

        """
        Load experiments data
        """
        if(draw_SGD_case):
            experiment = "SGD"
            sgd_data_path  = base_path + '/' + experiment + '/' + setting +".json"
            with open(sgd_data_path, "r") as data_file:
                data = json.load(data_file)
                SGD_train_accuracy = data["train_accuracy"]
                SGD_test_accuracy = data["test_accuracy"]
                SGD_epochs = len(SGD_train_accuracy)
                SGD_epoch_index = [i for i in range(1, SGD_epochs+1)]

        if(draw_DPSGD_BC_case):
            experiment = "SGD"
            # bc_data_path  = "./data/" + settings_path + '/' + experiment + '/' + setting +".json"
            bc_data_path  = base_path + '/' + experiment + '/BC/' + setting +".json"
            with open(bc_data_path, "r") as data_file:
                data = json.load(data_file)
                DPSGD_train_accuracy = data["train_accuracy"]
                DPSGD_test_accuracy = data["test_accuracy"]
                DPSGD_BC_epochs = len(DPSGD_train_accuracy)
                DPSGD_BC_epoch_index = [i for i in range(1, DPSGD_BC_epochs+1)]

        if(draw_DPSGD_IC_case):
            experiment = "SGD"
            ic_data_path = base_path + '/' + experiment + '/IC/' + setting +".json"
            with open(ic_data_path, "r") as data_file:
                data = json.load(data_file)
                IC_DPSGD_train_accuracy = data["train_accuracy"]
                IC_DPSGD_test_accuracy = data["test_accuracy"]
                DPSGD_IC_epochs = len(IC_DPSGD_train_accuracy)
                DPSGD_IC_epoch_index = [i for i in range(1, DPSGD_IC_epochs+1)]

        """
        Draw graphs
        """
        plt.subplot(1, 2, 1)
        if(draw_SGD_case):
            plt.plot(SGD_epoch_index, SGD_train_accuracy, label="SGD, s= %f" % (s))

        if(draw_DPSGD_BC_case):
            plt.plot(DPSGD_BC_epoch_index, DPSGD_train_accuracy, label="BC, s= %f" % (s))

        if(draw_DPSGD_IC_case):
            plt.plot(DPSGD_IC_epoch_index, IC_DPSGD_train_accuracy, label="IC, s= %f" % (s))
        plt.title('Train accuracy, lr = %f' % lr)
        plt.legend()

        plt.subplot(1, 2, 2)
        if(draw_SGD_case):
            plt.plot(SGD_epoch_index, SGD_test_accuracy, label="SGD, s= %f" % (s))
        if(draw_DPSGD_BC_case):
            plt.plot(DPSGD_BC_epoch_index, DPSGD_test_accuracy, label="BC,s= %f" % (s))
        if(draw_DPSGD_IC_case):
            plt.plot(DPSGD_IC_epoch_index, IC_DPSGD_test_accuracy, label="IC, s= %f" % (s))

        plt.title('Test accuracy, lr = %f, C = %f, sigma = %f' % (lr,C,sigma))
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        s = s*2
    if(draw_SGD_case and draw_DPSGD_BC_case):
        prefix = "SGD_BC"
    elif(draw_SGD_case and draw_DPSGD_IC_case):
        prefix = "SGD_IC"
    elif(draw_DPSGD_BC_case and draw_DPSGD_IC_case):
        prefix = "BC_IC"
    else:
        if (draw_SGD_case):
            prefix = "SGD"
        elif(draw_DPSGD_BC_case):
            prefix = "BC"
        else:
            prefix = "IC"
    file_name = '/' + prefix + '_lr_' + str(lr) + '_C_' + str(C) + '_sigma_' + str(sigma) + '_' + str(index)
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


