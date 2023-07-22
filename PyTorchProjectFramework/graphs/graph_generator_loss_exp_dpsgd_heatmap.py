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
import seaborn as sns
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
    settings_description = [
        {
            # Setting 0
            "settings_path": "settings_best_settings_lost_func",
            "Cs": [0.095]*30,
            "sigma": 0.01875,
            "lost_multis": [pow(2,i-15) for i in range(0,31)]
        },
        {
            # Setting 1
            "settings_path": "settings_best_settings_lost_func_new",
            "Cs": [2*0.095* ((i+10)//10) for i in range(0,31)],
            "sigma": 0.01875,
            "lost_multis": [pow(2,i%10-5) for i in range(0,31)]
        },
        {
            # Setting 2
            "settings_path": "settings_best_settings_lost_func_grid_search_1",
            "Cs": [0.01* ((i+10)//10) for i in range(0,31)],
            "sigma": 0.01875,
            "lost_multis": [pow(2,i%10-5) for i in range(0,31)]
        },
        {
            # Setting 3
            "settings_path": "settings_best_settings_lost_func_grid_search_2",
            "Cs": [0.08* ((i+10)//10) for i in range(0,31)],
            "sigma": 0.01875,
            "lost_multis": [pow(2,i%10-5) for i in range(0,31)]
        },
        {
            # Setting 4
            "settings_path": "settings_best_settings_lost_func_grid_search_3",
            "Cs": [0.64* ((i+10)//10) for i in range(0,31)],
            "sigma": 0.01875,
            "lost_multis": [pow(2,i%10-5) for i in range(0,31)]
        }

        # {
        #     # Setting 0
        #     "settings_path": "settings_sigma_dpsgd",
        #     "C": 1.2,
        #     "sigmas": [0.05*i for i in range(1,31)],
        #     "s": 64
        # }

    ]
    ############ TEST
    # np.random.seed(0)
    # sns.set()
    # uniform_data = np.random.rand(10, 12)
    # print(uniform_data.shape)
    #
    # ax = sns.heatmap(uniform_data, vmin=0, vmax=1)
    # plt.show()
    # input()

    ############
    # mode = None
    cmap = get_cmap(30)
    data_folder = "data_neurips"
    # mode = "shuffling"
    mode = "subsampling"
    clipping_mode = "layerwise"
    # clipping_mode = "all"
    # clipping_mode = "weight_FGC"
    # DGN = False
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
    is_constant_step_size =True
    # is_sigma_discounted = True
    models = ["Lenet","nor_Lenet" ,"convnet","nor_convnet","BNF_convnet", "AlexNet",
              "resnet18","resnet18_no_BN", "resnet34","resnet50","squarenet"]
    # Get models and settings
    setting_indexes = [2,3,4]# 0,3,6
    s_index =0
    models_index = 6
    model_name = models[models_index]
    heatmap_data = []
    C_data = []
    double_batch_size_settings = [i for i in range(1,31)]
    settings = ["setting_" + str(i) for i in double_batch_size_settings]
    lr = 0.025
    for setting_index in setting_indexes:
        print(type(setting_index))
        settings_path, Cs, sigma, lost_multis = settings_description[setting_index]["settings_path"], \
                                                settings_description[setting_index]["Cs"], \
                                                settings_description[setting_index]["sigma"], \
                                                settings_description[setting_index]["lost_multis"]
        graph_path = "./graph/" + settings_path + '/clipping'
        number_of_subgraphs = 1
        if (enable_mu):
            number_of_subgraphs = number_of_subgraphs + 1
        if(draw_training_acc):
            number_of_subgraphs = number_of_subgraphs + 1
        isExist = os.path.exists(graph_path)
        if (mode != None):
            base_path = "./" + data_folder + "/" + settings_path + "_" + mode
        else:
            base_path = "./" + data_folder + "/" + settings_path + "/" + model_name
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(graph_path)
            print("The new directory is created: %s" % graph_path)
        count = 0
        heatmap_count = 0
        sub_heatmap_data = [0]
        for setting_idx, setting in enumerate(settings):
            C = Cs[double_batch_size_settings[setting_idx]]
            heatmap_count = heatmap_count + 1
            if(heatmap_count % 10 == 0):
                heatmap_data.append(sub_heatmap_data)
                sub_heatmap_data = []
                C_data.append(C)
            print("Setting:", setting_idx+1)
            """
            Load experiments data
            """
            # if(draw_SGD_case and setting_idx == 0):
            if(draw_SGD_case):
                count = count +1
                cmap_color= cmap(4*count)
                experiment = "SGD"
                sgd_data_path = base_path + '/' + model_name + '/' + experiment + '/SGD/' + setting +".json"
                print(sgd_data_path)
                SGD_train_accuracy,SGD_test_accuracy = get_data(sgd_data_path)
                if(SGD_test_accuracy!= None):
                    epochs = len(SGD_test_accuracy)
                    print("SGD_test_accuracy",SGD_test_accuracy[-5:-1])

                    SGD_epochs = [i for i in range(1, epochs+1)]
                    print("SGD_epochs",SGD_epochs)
                    label = "SGD, No DP, loss_multi= %s" % (lost_multis[double_batch_size_settings[setting_idx]])
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
                label = "BC, ALC, loss_multi= %s, C= %s" % (lost_multis[double_batch_size_settings[setting_idx]],C)
                sub_heatmap_data.append(max(BC_DPSGD_test_accuracy[-5:-1])*100)
                # plt.plot(BC_epochs_idx, BC_DPSGD_test_accuracy, "o-", label=label, color=cmap_color,linewidth=3)


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
                label = "IC, ALC, loss_multi= %s" % (lost_multis[double_batch_size_settings[setting_idx]])
                plt.plot(IC_epochs_idx, IC_DPSGD_test_accuracy, "x--", label=label, color=cmap_color,linewidth=3)
        """
        Draw graphs
        """
    heatmap_data = np.array(heatmap_data)
    # import calendar
    # months = [month[:3] for month in calendar.month_name[1:]]
    sns.set()
    lost_multi_label = [pow(2,i%10-5) for i in range(0,11)]
    cmap = sns.color_palette("coolwarm", 128)
    ax = sns.heatmap(heatmap_data, vmin=0, vmax=100,
                     cmap=cmap,annot=True,
                     xticklabels=lost_multi_label, yticklabels=C_data)
    ax.set(xlabel='Loss_multi', ylabel='C')
    plt.show()

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

