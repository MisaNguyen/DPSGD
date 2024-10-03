import matplotlib.pyplot  as plt
import numpy as np

import os
import json
from scipy.interpolate import make_interp_spline
def get_data_from_settings(setting_info):
    """
    setting_info = {
        "data_folder": "data_neurips",
        "setting_path_base": "settings_clipping_exp_cifar10_dpsgd_large_C",
        "setting_index": "setting_30",
        "model_name": "convnet",
        "sampling_mode": "subsampling",
        "clipping_method": "IC",
        "clipping_mode": "layerwise",
        "DGN": True,
        "Optimizer": "SGD",
    }
    """
    if (setting_info["sampling_mode"] != None):
        base_path = "./" + setting_info["data_folder"] + "/" + setting_info["setting_path_base"] + "_" + setting_info["sampling_mode"]
    else:
        base_path = "./" + setting_info["data_folder"] + "/" + setting_info["setting_path_base"] + "/" +setting_info["model_name"]
    if (setting_info["clipping_method"] == None):
        data_path  = base_path +'/' + setting_info["model_name"] + '/' + \
                     setting_info["Optimizer"] + '/' + setting_info["Optimizer"] + '/' + setting_info["setting_index"] +".json"
    else:
        if(setting_info["DGN"]):
            data_path  = base_path + '_' + setting_info["clipping_method"]+'/' + setting_info["model_name"] + '/' + \
                            setting_info["Optimizer"] + '/' + setting_info["clipping_mode"] + '/' + setting_info["clipping_method"] \
                            + '/DGN/' + setting_info["setting_index"] +".json"
        elif(setting_info["AN"]):
            data_path  = base_path + '_' + setting_info["clipping_method"]+'/' + setting_info["model_name"] + '/' + \
                         setting_info["Optimizer"] + '/' + setting_info["clipping_mode"] + '/' + setting_info["clipping_method"] \
                         + '/AN/' + setting_info["setting_index"] +".json"
        else:
            data_path  = base_path + '_' + setting_info["clipping_method"]+'/' + setting_info["model_name"] + '/' + \
                            setting_info["Optimizer"] + '/' + setting_info["clipping_mode"] + '/' + setting_info["clipping_method"] \
                            + '/' + setting_info["setting_index"] +".json"
    isExist = os.path.exists(data_path)
    if isExist:
        with open(data_path, "r") as data_file:
            print("Loading setting:", data_path)
            data = json.load(data_file)
            DPSGD_train_accuracy = data["train_accuracy"]
            DPSGD_test_accuracy = data["test_accuracy"]
            DPSGD_epochs = len(DPSGD_train_accuracy)
    else:
        print("data_path not exist:", data_path)
        return None, None, None
    return DPSGD_train_accuracy, DPSGD_test_accuracy, DPSGD_epochs
def setup_plot(x_axis_label , y_axis_label,lr , C ):
    plt.subplot(1, 2, 1)
    plt.title('Train accuracy, lr = %f, C = %f' % (lr,C))
    plt.subplot(1, 2, 2)
    plt.title('Test accuracy, lr = %f, C = %f' % (lr,C))
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
def plt_draw(epoch_index, train_accuracy,test_accuracy,label,sigma,s):
    plt.subplot(1, 2, 1)
    # print(epoch_index)
    # print(train_accuracy)
    if (sigma!= None):
        plt.plot(epoch_index, train_accuracy, label=label % (sigma,s))
    else:
        plt.plot(epoch_index, train_accuracy, label=label)
    plt.subplot(1, 2, 2)
    if (sigma!= None):
        plt.plot(epoch_index, test_accuracy, label=label % (sigma,s))
    else:
        plt.plot(epoch_index, test_accuracy, label=label)


if __name__ == "__main__":
    # loading SGD data
    # settings_path = "settings_clipping_exp_cifar10_dpsgd_new"


    # setting_file_name = "settings_main_theorem(test)"
    # settings = ["setting_" + str(i) for i in range(1,6)]
    # settings = ["setting_" + str(i) for i in range(6,11)]
    # settings = ["setting_" + str(i) for i in range(11,16)]
    # settings = ["setting_" + str(i) for i in range(16,21)]

    # Cs = [0.1,0.05,0.01,0.005,0.5,1.0]
    # Cs = [1.0,1.5,2,2.5,3,3.5]
    # Cs = [6.0,7.0,8.0,9.0,10.0,20.0]

    # index=5
    # s_arr = [32,64,128,256,512]
    # s = s_arr[index-1]
    # C = 10
    # lr = 0.025
    # draw_IC_case = False
    # draw_BC_case = True
    # label = "BC sigma = %f, s = %f" if draw_BC_case else "IC sigma = %f, s = %f"
    # setup_plot('epoch' , 'accuracy',lr ,C)
    settings = ["setting_" + str(i) for i in range(13,25)]
    settings = ["setting_1"]
    test_acc_IC_DGN = []
    test_acc_IC_original = []
    test_acc_IC_AN = []
    for setting in settings:
        """Line1"""
        line_1_setting_info = {
            "data_folder": "data_neurips",
            "setting_path_base": "settings_duplicate",
            "setting_index": setting,
            "model_name": "convnet",
            "sampling_mode": "subsampling",
            "clipping_method": "IC",
            "clipping_mode": "layerwise",
            "DGN": True,
            "AN": False,
            "Optimizer": "SGD",
        }
        train_accuracy, test_accuracy, epochs = get_data_from_settings(line_1_setting_info)
        test_acc_IC_DGN.append(test_accuracy)

        """Line 4"""
        line_4_setting_info = {
            "data_folder": "data_neurips",
            "setting_path_base": "settings_duplicate",
            "setting_index": setting,
            "model_name": "convnet",
            "sampling_mode": "subsampling",
            "clipping_method": "IC",
            "clipping_mode": "layerwise",
            "DGN": False,
            "AN": True,
            "Optimizer": "SGD",
        }
        train_accuracy, test_accuracy, epochs = get_data_from_settings(line_4_setting_info)
        test_acc_IC_AN.append(test_accuracy)
        epoch_index = [i for i in range(1, epochs+1)]
        epoch_index = np.array(epoch_index)


    """ DRAW PLOT"""
    """Line1"""
    """IC + DGN + layerwise"""

    test_accuracy = np.array(np.mean(test_acc_IC_DGN, axis=0))

    # 500 represents number of points to make between T.min and T.max
    xnew = np.linspace(epoch_index.min(), epoch_index.max(), 500)
    X_Y_Spline = make_interp_spline(epoch_index, test_accuracy)
    test_accuracy_smooth = X_Y_Spline(xnew)

    label = "%s , %s, DGN: %s" % (line_1_setting_info["clipping_method"], line_1_setting_info["clipping_mode"]\
                                      ,str(line_1_setting_info["DGN"]))
    plt.plot(xnew,test_accuracy_smooth,label=label)

    # plt.plot(epoch_index, test_accuracy,label=label)


    """Line 2"""
    """IC + all"""
    line_2_setting_info = {
        "data_folder": "data_neurips_old",
        "setting_path_base": "settings_sigma_dpsgd_large_C",
        "setting_index": "setting_30",
        "model_name": "convnet",
        "sampling_mode": "subsampling",
        "clipping_method": "IC",
        "clipping_mode": "all",
        "DGN": False,
        "AN": False,
        "Optimizer": "SGD",
    }
    train_accuracy, test_accuracy, epochs = get_data_from_settings(line_2_setting_info)
    epoch_index = [i for i in range(1, epochs+1)]
    epoch_index = np.array(epoch_index)
    test_accuracy = np.array(test_accuracy)
    # 500 represents number of points to make between T.min and T.max
    xnew = np.linspace(epoch_index.min(), epoch_index.max(), 500)
    X_Y_Spline = make_interp_spline(epoch_index, test_accuracy)
    test_accuracy_smooth = X_Y_Spline(xnew)

    label = "%s , %s, DGN: %s" % (line_2_setting_info["clipping_method"], line_2_setting_info["clipping_mode"],\
                                  str(line_2_setting_info["DGN"]))
    plt.plot(xnew,test_accuracy_smooth,label=label)

    # plt.plot(epoch_index, test_accuracy,label=label)
    """Line 3"""
    """base_line"""
    line_3_setting_info = {
        "data_folder": "data_neurips",
        "setting_path_base": "settings_sigma_dpsgd",
        "setting_index": "setting_30",
        "model_name": "convnet",
        "sampling_mode": "subsampling",
        "clipping_method": None,
        "clipping_mode": "all",
        "DGN": False,
        "AN": False,
        "Optimizer": "SGD",
    }
    train_accuracy, test_accuracy, epochs = get_data_from_settings(line_3_setting_info)
    epoch_index = [i for i in range(1, epochs+1)]
    epoch_index = np.array(epoch_index)
    test_accuracy = np.array(test_accuracy)
    label = "Baseline"

    # 500 represents number of points to make between T.min and T.max
    xnew = np.linspace(epoch_index.min(), epoch_index.max(), 500)
    X_Y_Spline = make_interp_spline(epoch_index, test_accuracy)
    test_accuracy_smooth = X_Y_Spline(xnew)

    plt.plot(xnew,test_accuracy_smooth,label=label)
    # plt.show()
    # plt.plot(epoch_index, test_accuracy,label=label)
    # plt.plot(xnew, test_accuracy_smooth,label=label)
    plt.xlabel("epoch")
    plt.ylabel("Testing accuracy")
    plt.title("Layerwise versus Full gradient clipping")
    """Line 4"""
    # setting_info = {
    #     "data_folder": "data_neurips",
    #     "setting_path_base": "settings_sigma_dpsgd_large_C",
    #     "setting_index": "setting_30",
    #     "model_name": "convnet",
    #     "sampling_mode": "subsampling",
    #     "clipping_method": "IC",
    #     "clipping_mode": "layerwise",
    #     "DGN": False,
    #     "AN": True,
    #     "Optimizer": "SGD",
    # }
    test_accuracy = np.array(np.mean(test_acc_IC_AN, axis=0))
    xnew = np.linspace(epoch_index.min(), epoch_index.max(), 500)
    X_Y_Spline = make_interp_spline(epoch_index, test_accuracy)
    test_accuracy_smooth = X_Y_Spline(xnew)
    label = "IC, Zhang el at"
    # X_Y_Spline  = make_interp_spline(epoch_index, np.array(test_accuracy))
    # X_ = np.linspace(epoch_index.min(), epoch_index.max(), 500)
    plt.plot(epoch_index, test_accuracy,label=label)

    """Line 5"""
    line_5_setting_info = {
        "data_folder": "data_neurips",
        "setting_path_base": "settings_sigma_dpsgd",
        "setting_index": "setting_30",
        "model_name": "resnet18",
        "sampling_mode": "subsampling",
        "clipping_method": "IC",
        "clipping_mode": "layerwise",
        "DGN": False,
        "AN": True,
        "Optimizer": "SGD",
    }
    train_accuracy, test_accuracy, epochs = get_data_from_settings(line_5_setting_info)
    epoch_index = [i for i in range(1, epochs+1)]
    epoch_index = np.array(epoch_index)

    xnew = np.linspace(epoch_index.min(), epoch_index.max(), 500)
    X_Y_Spline = make_interp_spline(epoch_index, test_accuracy)
    test_accuracy_smooth = X_Y_Spline(xnew)
    label = "IC, Zhang el at, sigma = 60, resnet18"
    # X_Y_Spline  = make_interp_spline(epoch_index, np.array(test_accuracy))
    # X_ = np.linspace(epoch_index.min(), epoch_index.max(), 500)
    plt.plot(epoch_index, test_accuracy,label=label)

    """Line 6"""
    line_6_setting_info = {
        "data_folder": "data_neurips",
        "setting_path_base": "settings_sigma_dpsgd",
        "setting_index": "setting_30",
        "model_name": "resnet18",
        "sampling_mode": "subsampling",
        "clipping_method": "IC",
        "clipping_mode": "layerwise",
        "DGN": True,
        "AN": False,
        "Optimizer": "SGD",
    }
    train_accuracy, test_accuracy, epochs = get_data_from_settings(line_6_setting_info)
    epoch_index = [i for i in range(1, epochs+1)]
    epoch_index = np.array(epoch_index)

    xnew = np.linspace(epoch_index.min(), epoch_index.max(), 500)
    X_Y_Spline = make_interp_spline(epoch_index, test_accuracy)
    test_accuracy_smooth = X_Y_Spline(xnew)
    label = "IC, layerwise,DGN, sigma = 60, resnet18"
    # X_Y_Spline  = make_interp_spline(epoch_index, np.array(test_accuracy))
    # X_ = np.linspace(epoch_index.min(), epoch_index.max(), 500)
    plt.plot(epoch_index, test_accuracy,label=label)

    """Line 7"""
    line_7_setting_info = {
        "data_folder": "data_neurips",
        "setting_path_base": "settings_sigma_dpsgd",
        "setting_index": "setting_30",
        "model_name": "resnet18",
        "sampling_mode": "subsampling",
        "clipping_method": "BC",
        "clipping_mode": "layerwise",
        "DGN": True,
        "AN": False,
        "Optimizer": "SGD",
    }
    train_accuracy, test_accuracy, epochs = get_data_from_settings(line_7_setting_info)
    epoch_index = [i for i in range(1, epochs+1)]
    epoch_index = np.array(epoch_index)

    xnew = np.linspace(epoch_index.min(), epoch_index.max(), 500)
    X_Y_Spline = make_interp_spline(epoch_index, test_accuracy)
    test_accuracy_smooth = X_Y_Spline(xnew)
    label = "BC, layerwise,DGN, sigma = 60, resnet18"
    # X_Y_Spline  = make_interp_spline(epoch_index, np.array(test_accuracy))
    # X_ = np.linspace(epoch_index.min(), epoch_index.max(), 500)
    plt.plot(epoch_index, test_accuracy,label=label)
    # plt.xlabel("epoch")
    # plt.ylabel("Testing accuracy")


    """----------------------------------------------"""
    plt.title("Layerwise versus Full gradient clipping")
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches((22, 11), forward=False)
    graph_path = "./graph/sigma_custom_compare"
    file_name ="/LayerwiseVSFull"
    # Check whether the specified path exists or not
    isExist = os.path.exists(graph_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(graph_path)
        print("The new directory is created: %s" % graph_path)
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


