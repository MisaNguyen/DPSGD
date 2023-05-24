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
    settings_info = [
        {
            # Setting 0
            "settings_path": "settings_sigma_dpsgd",
            "C": 0.005,
            "sigmas": [2*i for i in range(1,31)],
            "s": 64
        },
        {
            # Setting 1
            "settings_path": "settings_sigma_dpsgd_large_sigma",
            "C": 0.005,
            "sigmas": [64*i for i in range(1,31)],
            "s": 64
        },
        {
            # Setting 2
            "settings_path": "settings_sigma_dpsgd_super_sigma",
            "C": 0.005,
            "sigmas": [1024*pow(2,i) for i in range(0,31)],
            "s": 64
        },
        {
            # Setting 3
            "settings_path": "settings_sigma_dpsgd_final",
            "C": 0.005,
            "sigmas": [2*pow(2,i) for i in range(0,31)],
            "s": 64
        },
        {
            # Setting 4
            "settings_path": "settings_sigma_dpsgd_small_C",
            "C": 0.0005,
            "sigmas": [2*pow(2,i) for i in range(0,31)],
            "s": 64
        },
        {
            # Setting 5
            "settings_path": "settings_sigma_dpsgd_super_small_C",
            "C": 0.00005,
            "sigmas": [2*pow(2,i) for i in range(0,31)],
            "s": 64
        },
        {
            # Setting 6
            "settings_path": "settings_vary_sigma",
            "C": 0.95,
            "sigmas": [0.15*pow(1/2,i) for i in range(0,31)],
            "s": 64
        },
        {
            # Setting 7
            "settings_path": "settings_vary_sigma_resnet18",
            "C": 0.095,
            "sigmas": [0.01875*pow(2,i) for i in range(0,31)],
            "s": 64
        },
        {
            # Setting 8
            "settings_path": "settings_vary_sigma_LeNet",
            "C": 0.2,
            "sigmas": [0.1 + 0.1*i for i in range(0,31)],
            "s": 64
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
    clipping_mode = "layerwise"
    # clipping_mode = "all"
    DGN = False
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
    models = ["Lenet","nor_Lenet", "convnet","nor_convnet","BNF_convnet", "AlexNet",
              "resnet18", "resnet34","resnet50","squarenet"]
    # Get models and settings
    # setting_indexes = [3,4,5] # 0,3,6
    setting_indexes = [8] # 0,3,6
    models_index = 1
    model_name =models[models_index]
    count = 0
    for idx, setting_index in enumerate(setting_indexes):
        # s_index =0
        # print(setting_index)
        # print(type(setting_index))
        # print(settings[setting_index])
        # input()
        settings_path, C, sigmas, s = settings_info[setting_index]["settings_path"], \
                                      settings_info[setting_index]["C"], \
                                      settings_info[setting_index]["sigmas"], \
                                      settings_info[setting_index]["s"]
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
        settings = ["setting_" + str(i) for i in range(1,30)]
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

        sgd_testing_acc = []
        IC_testing_acc=[]
        BC_testing_acc=[]
        AN_BC_testing_acc=[]
        sgd_sigma_arr = []
        IC_sigma_arr = []
        BC_sigma_arr = []
        AN_BC_sigma_arr = []
        for setting_idx, setting in enumerate(settings):

            # if (setting == "setting_30"):
            #     s=512
            print("Setting:", setting_idx)
            """
            Load experiments data
            """
            if(draw_SGD_case):
                experiment = "SGD"
                # sgd_data_path  = base_path + '/' + experiment + '/' + setting +".json"
                sgd_data_path = base_path + '/' + model_name + '/' + experiment + '/SGD/' + setting +".json"
                print(sgd_data_path)
                SGD_train_accuracy,SGD_test_accuracy = get_data(sgd_data_path)
                if(SGD_test_accuracy!= None):
                    sgd_testing_acc.append(max(SGD_test_accuracy[-5:-1]))
                    sgd_sigma_arr.append(sigmas[setting_idx])
                    epochs = len(SGD_train_accuracy)
                # DPSGD_SGD_epoch_index = [i for i in range(1, DPSGD_SGD_epochs+1)]

            if(draw_DPSGD_BC_case):
                experiment = "SGD"
                bc_data_path = base_path + '_BC/' + model_name + '/' + experiment \
                               + '/' + clipping_mode
                # bc_data_path  = "./data/" + settings_path + '/' + experiment + '/' + setting +".json"
                # if(constant_ci):
                #     ic_data_path  = bc_data_path + "/constant_c_i"


                if(DGN):
                    bc_data_path  = bc_data_path + '/BC/DGN/'
                else:
                    bc_data_path  = bc_data_path + '/BC/'
                if (draw_AN and idx <1):
                    AN_bc_data_path = bc_data_path + 'AN/'
                    # print("DEBUG",AN_bc_data_path)
                    AN_bc_data_path  = AN_bc_data_path + setting +".json"
                    AN_BC_DPSGD_train_accuracy,AN_BC_DPSGD_test_accuracy = get_data(AN_bc_data_path)

                    if(AN_BC_DPSGD_test_accuracy!= None):
                        AN_BC_testing_acc.append(max(AN_BC_DPSGD_test_accuracy[-5:-1]))
                        AN_BC_sigma_arr.append(sigmas[setting_idx])
                        epochs = len(AN_BC_DPSGD_test_accuracy)
                bc_data_path  = bc_data_path + setting +".json"
                # bc_data_path  = base_path + '_BC/' + model_name + '/' + experiment \
                #                  + '/BC/' + setting +".json"
                print("bc_data_path:",bc_data_path)
                BC_DPSGD_train_accuracy,BC_DPSGD_test_accuracy = get_data(bc_data_path)
                if(BC_DPSGD_test_accuracy!= None):
                    BC_testing_acc.append(max(BC_DPSGD_test_accuracy[-5:-1]))
                    BC_sigma_arr.append(sigmas[setting_idx])
                    epochs = len(BC_DPSGD_train_accuracy)
                # DPSGD_BC_epoch_index = [i for i in range(1, DPSGD_BC_epochs+1)]
            # else:
            #     s = s*2
            #     continue


            if(draw_DPSGD_IC_case):
                experiment = "SGD"
                # ic_data_path = base_path + '_IC/' + model_name + '/' + experiment + '/IC/' + setting +".json"
                ic_data_path  = base_path + '_IC/' + model_name + '/' + experiment \
                                + '/' + clipping_mode
                if(constant_ci):
                    ic_data_path  = ic_data_path + "/constant_c_i"
                if(DGN):
                    ic_data_path  = ic_data_path + '/IC/DGN/' + setting +".json"
                else:
                    ic_data_path  = ic_data_path + '/IC/' + setting +".json"
                print(ic_data_path)
                IC_DPSGD_train_accuracy,IC_DPSGD_test_accuracy = get_data(ic_data_path)
                if(IC_DPSGD_test_accuracy!= None):
                    IC_testing_acc.append(max(IC_DPSGD_test_accuracy[-5:-1]))
                    IC_sigma_arr.append(sigmas[setting_idx])
                    epochs = len(IC_DPSGD_train_accuracy)
                # DPSGD_IC_epoch_index = [i for i in range(1, DPSGD_IC_epochs+1)]

            # if(draw_mixing_case):
            #     experiment = "SGD"
            #     ic_data_path = base_path + '/' + model_name + '/' + experiment + '/NM/' + setting +".json"
            #     with open(ic_data_path, "r") as data_file:
            #         data = json.load(data_file)
            #         Mixing_DPSGD_train_accuracy = data["train_accuracy"]
            #         Mixing_DPSGD_test_accuracy = data["test_accuracy"]
            #         DPSGD_Mixing_epochs = len(Mixing_DPSGD_train_accuracy)
            #         DPSGD_Mixing_epoch_index = [i for i in range(1, DPSGD_Mixing_epochs+1)]

            """
            Draw graphs
            """

        sigma_best_index = BC_testing_acc.index(max(BC_testing_acc))
        sigma_best_max = sigmas[sigma_best_index]
        print("BC_testing_acc", BC_testing_acc[-5:-1])
        print("BC_sigma_arr", BC_sigma_arr[-5:-1])
        print(sigma_best_max)
        # if(draw_training_acc):
        #
        #     plt.subplot(1, number_of_subgraphs, 1)
        #     plt.title('Train accuracy, lr = %f' % lr)
        #
        if(draw_SGD_case):
            count = count +1
            cmap_color = cmap(4*count)
            plt.plot(sgd_sigma_arr, sgd_testing_acc, label="SGD", color=cmap_color)
            # mu = [0 for E in SGD_epoch_index]
            print("SGD training acc:", SGD_train_accuracy[-1])
        if(draw_AN and idx <1):
            count = count +1
            cmap_color = cmap(4*count)
            # label = "BC, %s" % ( clipping_mode)
            label = "Zhang el. at"
            # if (DGN):
            #     label = label + ", diminishing C"
            plt.plot(AN_BC_sigma_arr, AN_BC_testing_acc, "o-", label=label, color='blue',linewidth=3)
            plt.subplot(1,number_of_subgraphs,number_of_subgraphs)
        if(draw_DPSGD_BC_case):
            count = count +1
            cmap_color = cmap(4*count)
            # label = "BC, %s" % ( clipping_mode)
            # label = "BC,ALC, C= %s" % (C)
            label = "BC, ALC"
            # if (DGN):
            #     label = label + ", diminishing C"
            plt.plot(BC_sigma_arr, BC_testing_acc, "o-", label=label, color="black",linewidth=3)
            plt.subplot(1,number_of_subgraphs,number_of_subgraphs)
            # mu = [np.sqrt(E)/sigma for E in DPSGD_BC_epoch_index]
            # mu_index = DPSGD_BC_epoch_index
            # print("BC training acc:", BC_DPSGD_train_accuracy[-1])

        if(draw_DPSGD_IC_case):
            print(IC_testing_acc)
            count = count +1
            cmap_color = cmap(4*count)
            # label = "IC, %s" % ( clipping_mode)
            label = "IC"
            # if (DGN):
            #     label = label + ", diminishing C"
            plt.plot(IC_sigma_arr, IC_testing_acc, "x--", label=label , color='red',linewidth=3)
            plt.subplot(1,number_of_subgraphs,number_of_subgraphs)
        # mu = [np.sqrt(E)/sigma for E in DPSGD_IC_epoch_index]
        # mu_index = DPSGD_IC_epoch_index
        # print("IC training acc:", IC_DPSGD_train_accuracy[-1])
        # experiment = "SGD"
    plt.xlabel("$\sigma$")
    plt.ylabel("accuracy")
    plt.title(" $\eta =$ %s, m = %s, C = %s, E =%s" % ( lr,s,C,epochs))
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


