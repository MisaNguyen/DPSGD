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
            "settings_path": "settings_best_settings",
            "C": 0.095,
            "sigma": 0.01875,
            "ss": [64*i for i in range(0,31)]
        },
        {
            # Setting 1
            "settings_path": "settings_best_settings_convnet",
            "C": 0.14,
            "sigma": 0.5,
            "ss": [0+ 64*i for i in range(0,31)]
        },
        {
            # Setting 2
            "settings_path": "settings_best_settings_LeNet",
            "C": 0.2,
            "sigma": 2.5,
            "ss": [64*i for i in range(0,31)]
        },
        {
            # Setting 3
            "settings_path": "settings_best_settings_new",
            "C": 2*0.095,
            "sigma": 0.01875,
            "ss": [64*i for i in range(0,31)]
        }

        # {
        #     # Setting 0
        #     "settings_path": "settings_sigma_dpsgd",
        #     "C": 1.2,
        #     "sigmas": [0.05*i for i in range(1,31)],
        #     "s": 64
        # }

    ]
    # mode = None
    cmap = get_cmap(30)
    data_folder = "data_neurips"
    # mode = "shuffling"
    mode = "subsampling"
    # clipping_mode = "layerwise"
    # clipping_mode = "all"
    clipping_mode = "weight_FGC"
    # DGN = False
    DGN = True
    print("DGN:", DGN)
    # DGN = None
    # clipping_mode = ""
    draw_DPSGD_IC_case = False
    draw_SGD_case = False
    draw_DPSGD_BC_case = True
    draw_mixing_case = False
    enable_mu = False
    draw_training_acc = False
    constant_ci = False
    draw_AN = False #Zhang setting
    is_constant_step_size =True
    # is_sigma_discounted = True
    models = ["Lenet","nor_Lenet" ,"convnet","nor_convnet","BNF_convnet", "AlexNet",
              "resnet18","resnet18_no_BN", "resnet34","resnet50","squarenet"]
    # Get models and settings
    setting_index = 3# 0,3,6
    s_index =0
    # models_index = 7
    # models_index = 2
    models_index = 6
    model_name = models[models_index]
    settings_path, C, sigma, ss = settings[setting_index]["settings_path"], \
                                        settings[setting_index]["C"], \
                                        settings[setting_index]["sigma"], \
                                        settings[setting_index]["ss"]

    # Partition setting
    partition = False


    # setting_file_name = "settings_main_theorem(test)"
    # settings = ["setting_" + str(i) for i in range(1,6)]
    # settings = ["setting_" + str(i) for i in range(6,11)]
    # settings = ["setting_" + str(i) for i in range(11,16)]
    # settings = ["setting_" + str(i) for i in range(16,21)]
    # settings = ["setting_" + str(i) for i in range(21,26)]
    # settings = ["setting_" + str(i) for i in range(26,31)]

    # s_index_min = 1 # min = 1
    # s_index_max = 6 # max = 6
    # setting_index = 1
    # settings = ["setting_" + str(setting_index)]
    # settings = ["setting_" + str(i) for i in range(1,11)]
    double_batch_size_settings = [1,2,4,8,16]
    # double_batch_size_settings = [1]
    # double_batch_size_settings = [1,3,7,15]
    settings = ["setting_" + str(i) for i in double_batch_size_settings]
    # settings.append("setting_30")
    # settings = ["setting_" + str(5*s_index+i) for i in range(s_index_min,s_index_max)]
    # settings = ["setting_0" ]
    lr = 0.025

    # Cs = [0.1,0.05,0.01,0.005,0.5,1.0] #old
    # Cs = [1.0,1.5,2,2.5,3,3.5]
    # Cs = [6.0,7.0,8.0,9.0,10.0,20.0]
    # C = Cs[s_index]
    # s = s_start * pow(2, s_index_min-1)

    # settings = ["setting_0_c1_s2","setting_0_noclip"]
    # settings = ["setting_1","setting_2","setting_3","setting_4"]
    # settings = ["setting_1","setting_2"]
    # settings = ["setting_5","setting_6","setting_7","setting_8"]
    # settings = ["setting_0"]
    graph_path = "./graph/" + settings_path + '/clipping'

    number_of_subgraphs = 1
    if (enable_mu):
        number_of_subgraphs = number_of_subgraphs + 1
    if(draw_training_acc):
        number_of_subgraphs = number_of_subgraphs + 1
    # if(draw_DPSGD_BC_case and draw_DPSGD_IC_case):
    #     number_of_subgraphs = number_of_subgraphs + 1
    # Check whether the specified path exists or not
    isExist = os.path.exists(graph_path)
    if (mode != None):
        base_path = "./" + data_folder + "/" + settings_path + "_" + mode
    else:
        base_path = "./" + data_folder + "/" + settings_path + "/" + model_name
    # base_path = "./data/" + settings_path + "/" + model_name
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(graph_path)
        print("The new directory is created: %s" % graph_path)
    count = 0

    for setting_idx, setting in enumerate(settings):

        # if (setting == "setting_30"):
        #     s=512
        print("Setting:", setting_idx)
        """
        Load experiments data
        """
        # if(draw_SGD_case and setting_idx == 0):
        if(draw_SGD_case):
            count = count +1
            cmap_color= cmap(4*count)
            experiment = "SGD"
            # sgd_data_path  = base_path + '/' + experiment + '/' + setting +".json"
            sgd_data_path = base_path + '/' + model_name + '/' + experiment + '/SGD/' + setting +".json"
            print(sgd_data_path)
            SGD_train_accuracy,SGD_test_accuracy = get_data(sgd_data_path)
            if(SGD_test_accuracy!= None):
                epochs = len(SGD_test_accuracy)
                print("SGD_test_accuracy",SGD_test_accuracy[-5:-1])

                SGD_epochs = [i for i in range(1, epochs+1)]
                print("SGD_epochs",SGD_epochs)
                label = "SGD, No DP, m= %s" % (ss[double_batch_size_settings[setting_idx]])
                plt.plot(SGD_epochs, SGD_test_accuracy, "o-", label=label, color=cmap_color,linewidth=3)
                # print("SGD_test_accuracy", SGD_test_accuracy[-5])
            # DPSGD_SGD_epoch_index = [i for i in range(1, DPSGD_SGD_epochs+1)]
        if(draw_AN):
            count = count +1
            cmap_color= cmap(4*count)
            experiment = "SGD"
            # bc_data_path  = "./data/" + settings_path + '/' + experiment + '/' + setting +".json"
            Zhang_data_path  = base_path + '_BC/' + model_name + '/' + experiment \
                            + '/' + clipping_mode + '/BC/AN/' + setting +".json"
            # bc_data_path  = base_path + '_BC/' + model_name + '/' + experiment \
            #                  + '/BC/' + setting +".json"
            print("Zhang_data_path:",Zhang_data_path)
            zhang_DPSGD_train_accuracy,zhang_DPSGD_test_accuracy = get_data(Zhang_data_path)
            print("BC_DPSGD_test_accuracy",zhang_DPSGD_test_accuracy[-5:-1])
            epochs = len(zhang_DPSGD_test_accuracy)
            zhang_epochs_idx = [i for i in range(1, epochs+1)]
            # label = "BC, %s" % ( clipping_mode)
            label = "Zhang el at"
            # if (DGN):
            #     label = label + ", diminishing C"
            plt.plot(zhang_epochs_idx, zhang_DPSGD_test_accuracy, "o-", label=label, color="blue",linewidth=3)
            # DPSGD_BC_epoch_index = [i for i in range(1, DPSGD_BC_epochs+1)]
        if(draw_DPSGD_BC_case):
            count = count +1
            cmap_color= cmap(4*count)
            experiment = "SGD"
            # bc_data_path  = "./data/" + settings_path + '/' + experiment + '/' + setting +".json"
            if(is_constant_step_size):
                bc_data_path = base_path + "_BC_css/"
            else:
                bc_data_path = base_path + '_BC/'
            if(DGN):
                bc_data_path  = bc_data_path + model_name + '/' + experiment \
                                + '/' + clipping_mode + '/BC/DGN/' + setting +".json"
            else:
                bc_data_path  = bc_data_path + model_name + '/' + experiment \
                                + '/' + clipping_mode + '/BC/' + setting +".json"
            # bc_data_path  = base_path + '_BC/' + model_name + '/' + experiment \
            #                  + '/BC/' + setting +".json"
            print("bc_data_path:",bc_data_path)
            BC_DPSGD_train_accuracy,BC_DPSGD_test_accuracy = get_data(bc_data_path)
            print("BC_DPSGD_test_accuracy",BC_DPSGD_test_accuracy[-5:-1])
            epochs = len(BC_DPSGD_train_accuracy)
            BC_epochs_idx = [i for i in range(1, epochs+1)]
            label = "BC, ALC, m= %s" % (ss[double_batch_size_settings[setting_idx]])
            plt.plot(BC_epochs_idx, BC_DPSGD_test_accuracy, "o-", label=label, color=cmap_color,linewidth=3)

            # """ TEMP"""
            # clipping_mode= "all"
            # count = count +1
            # cmap_color= cmap(4*count)
            # experiment = "SGD"
            # # bc_data_path  = "./data/" + settings_path + '/' + experiment + '/' + setting +".json"
            # if(DGN):
            #     bc_data_path  = base_path + '_BC/' + model_name + '/' + experiment \
            #                     + '/' + clipping_mode + '/BC/DGN/discounted/' + setting +".json"
            # else:
            #     bc_data_path  = base_path + '_BC/' + model_name + '/' + experiment \
            #                     + '/' + clipping_mode + '/BC/discounted/' + setting +".json"
            # # bc_data_path  = base_path + '_BC/' + model_name + '/' + experiment \
            # #                  + '/BC/' + setting +".json"
            # print("bc_data_path:",bc_data_path)
            # BC_DPSGD_train_accuracy,BC_DPSGD_test_accuracy = get_data(bc_data_path)
            # print("BC_DPSGD_test_accuracy",BC_DPSGD_test_accuracy[-5:-1])
            # epochs = len(BC_DPSGD_train_accuracy)
            # BC_epochs_idx = [i for i in range(1, epochs+1)]
            # label = "BC, FGC, m= %s, $\\bar{\sigma} =\sigma/62$" % (ss[double_batch_size_settings[setting_idx]])
            # plt.plot(BC_epochs_idx, BC_DPSGD_test_accuracy, "o-", label=label, color=cmap_color,linewidth=3)
            # """ END TEMP """
            # DPSGD_BC_epoch_index = [i for i in range(1, DPSGD_BC_epochs+1)]
        # else:
        #     s = s*2
        #     continue


        if(draw_DPSGD_IC_case):
            count = count +1
            cmap_color= cmap(4*count)
            experiment = "SGD"
            # ic_data_path = base_path + '_IC/' + model_name + '/' + experiment + '/IC/' + setting +".json"
            if(is_constant_step_size):
                ic_data_path = base_path + "_IC_css/"
            else:
                ic_data_path = base_path + '_IC/'
            ic_data_path  = ic_data_path + model_name + '/' + experiment \
                            + '/' + clipping_mode

            if(constant_ci):
                ic_data_path  = ic_data_path + "/constant_c_i"
            if(DGN):
                ic_data_path  = ic_data_path + '/IC/DGN/' + setting +".json"
            else:
                ic_data_path  = ic_data_path + '/IC/' + setting +".json"
            print(ic_data_path)
            IC_DPSGD_train_accuracy,IC_DPSGD_test_accuracy = get_data(ic_data_path)
            print("IC_DPSGD_test_accuracy",IC_DPSGD_test_accuracy[-5:-1])
            epochs = len(IC_DPSGD_train_accuracy)
            IC_epochs_idx = [i for i in range(1, epochs+1)]
            # label = "IC, %s" % ( clipping_mode)
            label = "IC, ALC, m= %s" % (ss[double_batch_size_settings[setting_idx]])
            # label = "IC"
            # if (DGN):
            #     label = label + ", diminishing C"
            plt.plot(IC_epochs_idx, IC_DPSGD_test_accuracy, "x--", label=label, color=cmap_color,linewidth=3)
            # plt.plot(IC_epochs_idx, IC_DPSGD_test_accuracy, "x--", label=label, color="red",linewidth=3)
            # DPSGD_IC_epoch_index = [i for i in range(1, DPSGD_IC_epochs+1)]


        """
        Draw graphs
        """
    # count = 5

    # if(draw_training_acc):
    #
    #     plt.subplot(1, number_of_subgraphs, 1)
    #     plt.title('Train accuracy, lr = %f' % lr)
    #
        # plt.subplot(1,number_of_subgraphs,number_of_subgraphs)
        # # mu = [np.sqrt(E)/sigma for E in DPSGD_IC_epoch_index]
        # # mu_index = DPSGD_IC_epoch_index
        # print("IC training acc:", IC_DPSGD_train_accuracy[-1])
        # experiment = "SGD"
    plt.xlabel("Epoch")
    plt.ylabel("accuracy")

    plt.title("$\eta =$ %s, C = %s, $\sigma=%s$, E =%s" %  (lr,C,sigma,epochs))
    # plt.title("$\eta =$ %s, E =%s" %  (lr,epochs))
    plt.legend()
        # ic_data_path = base_path + '_IC/' + model_name + '/' + experiment + '/IC/' + setting +".json"
        #
        #
        #     if(draw_mixing_case):
        #         plt.plot(DPSGD_Mixing_epoch_index, Mixing_DPSGD_train_accuracy, "x-", label="mixing, m= %f" % (s), color=cmap_color)
        #         plt.subplot(1,number_of_subgraphs,number_of_subgraphs)
        #         mu = [np.sqrt(E)/sigma for E in DPSGD_Mixing_epoch_index]
        #         mu_index = DPSGD_Mixing_epoch_index
        #         print("Mixing training acc:", Mixing_DPSGD_train_accuracy[-1])
        #     plt.legend()
        # # input(number_of_subgraphs)
        # if(draw_training_acc):
        #     plt.subplot(1, number_of_subgraphs, 2)
        # else:
        #     plt.subplot(1, number_of_subgraphs, 1)
        # if(draw_mixing_case or draw_DPSGD_IC_case or draw_DPSGD_BC_case):
        #     plt.title('Test accuracy, lr = %s, C = %s, sigma = %d' % (str(lr).strip('0'),str(C).strip('0'),sigma))
        # else:
        #     plt.title('Test accuracy, lr = %s' % (str(lr).strip('0')))
        # if(draw_SGD_case):
        #     plt.plot(SGD_epoch_index, SGD_test_accuracy, label="SGD, s= %d" % (s), color=cmap_color)
        #     print("SGD test acc:", SGD_test_accuracy[-5:-1])
        # if(draw_DPSGD_BC_case):
        #     plt.plot(DPSGD_BC_epoch_index, BC_DPSGD_test_accuracy, "o-", label="BC,", color=cmap_color)
        #     print("BC test acc:", BC_DPSGD_test_accuracy[-5:-1])
        # if(draw_DPSGD_IC_case):
        #     print("HERE")
        #     plt.plot(DPSGD_IC_epoch_index, IC_DPSGD_test_accuracy, "x-",label="IC, m= %d, %s" % (s,clipping_mode), color=cmap_color)
        #     print("IC test acc:", IC_DPSGD_test_accuracy[-5:-1])
        #     # clipping_mode = "all"
        #     # if(DGN):
        #     #     ic_data_path  = base_path + '_IC/' + model_name + '/' + experiment \
        #     #                     + '/' + clipping_mode + '/IC/DGN/' + setting +".json"
        #     # else:
        #     #     ic_data_path  = base_path + '_IC/' + model_name + '/' + experiment \
        #     #                     + '/' + clipping_mode + '/IC/' + setting +".json"
        #     # print(ic_data_path)
        #     # if (os.path.exists(ic_data_path)):
        #     #     print("here")
        #     #     with open(ic_data_path, "r") as data_file:
        #     #         data = json.load(data_file)
        #     #         IC_DPSGD_train_accuracy = data["train_accuracy"]
        #     #         IC_DPSGD_test_accuracy = data["test_accuracy"]
        #     #         DPSGD_IC_epochs = len(IC_DPSGD_train_accuracy)
        #     #         DPSGD_IC_epoch_index = [i for i in range(1, DPSGD_IC_epochs+1)]
        #     # cmap_color = cmap(4*setting_idx+1)
        #     # plt.plot(DPSGD_IC_epoch_index, IC_DPSGD_test_accuracy, "o-",label="IC, m= %d, %s" % (s,clipping_mode), color=cmap_color)
        #
        # if(draw_mixing_case):
        #     plt.plot(DPSGD_Mixing_epoch_index, Mixing_DPSGD_test_accuracy, "x-",label="MC, ", color=cmap_color)
        #     print("Mixing test acc:", Mixing_DPSGD_test_accuracy[-5:-1])
        # plt.legend()
        # if(enable_mu):
        #     if(draw_DPSGD_BC_case or draw_DPSGD_IC_case or draw_mixing_case):
        #         plt.subplot(1,number_of_subgraphs,3)
        #         plt.plot(mu_index, mu, label="mu per epoch", color=cmap_color)
        #         plt.legend()
        # if(draw_DPSGD_BC_case and draw_DPSGD_IC_case):
        #     plt.subplot(1,number_of_subgraphs,4)
        #     test_acc_ratio = [BC_DPSGD_test_accuracy[i]/ IC_DPSGD_test_accuracy[i] for i in range(len(DPSGD_test_accuracy))]
        #     plt.plot(DPSGD_BC_epoch_index, test_acc_ratio, label="BC/IC,s= %f" % (s), color=cmap_color)
        #     plt.legend()
        # plt.xlabel('epoch')
        # plt.ylabel('accuracy')
        # s = s*2
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
    if(draw_mixing_case):
        prefix = "MC"
    fig_index = setting_index


    file_name =  '/' + model_name + '_'+ mode + '_' + prefix + '_lr_' + str(lr) + '_C_' + str(C)  + '_' + str(fig_index)
    while(os.path.exists(graph_path+ file_name)):
        fig_index = fig_index+1
        file_name =  '/' + model_name + '_' + mode + '_' + prefix + '_lr_' + str(lr) + '_C_' + str(C) + '_' + str(fig_index)
    fig = plt.gcf()
    fig.set_size_inches((22, 11), forward=False)
    print("saving to ", graph_path + file_name +".png")
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


