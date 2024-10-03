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
        "constant_step_size": True,
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
        if (setting_info["constant_step_size"] == True):
            base_path = base_path + '_' + setting_info["clipping_method"] + "_" + "css"
        else:
            base_path = base_path + '_' + setting_info["clipping_method"]
        if(setting_info["DGN"]):
            data_path  = base_path +'/' + setting_info["model_name"] + '/' + \
                            setting_info["Optimizer"] + '/' + setting_info["clipping_mode"] + '/' + setting_info["clipping_method"] \
                            + '/DGN/' + setting_info["setting_index"] +".json"
        elif(setting_info["AN"]):
            data_path  = base_path +'/' + setting_info["model_name"] + '/' + \
                         setting_info["Optimizer"] + '/' + setting_info["clipping_mode"] + '/' + setting_info["clipping_method"] \
                         + '/AN/' + setting_info["setting_index"] +".json"
        else:
            data_path  = base_path +'/' + setting_info["model_name"] + '/' + \
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

    # settings = ["setting_1"]
    setting = "setting_1"
    test_acc_IC_LeNet_layerwise = []
    test_acc_IC_LeNet_layerwise_css = []
    test_acc_BC_LeNet_layerwise = []
    test_acc_BC_LeNet_layerwise_css = []
    test_acc_BC_LeNet_layerwise_BN = []
    test_acc_BC_LeNet_layerwise_css_BN = []
    test_acc_baseline = []

    """Line 1"""
    line_1_setting_info = {
        "data_folder": "data_neurips",
        "setting_path_base": "settings_clipping_exp_cifar10_dpsgd_opacus",
        "setting_index": setting,
        "model_name": "LeNet",
        "sampling_mode": "subsampling",
        "clipping_method": "IC",
        "clipping_mode": "layerwise",
        "constant_step_size": False,
        "DGN": False,
        "AN": False,
        "Optimizer": "SGD",
    }
    train_accuracy, test_accuracy, epochs = get_data_from_settings(line_1_setting_info)
    test_acc_IC_LeNet_layerwise.append(test_accuracy)


    """Line 2"""
    line_2_setting_info = {
        "data_folder": "data_neurips",
        "setting_path_base": "settings_clipping_exp_cifar10_dpsgd_opacus",
        "setting_index": setting,
        "model_name": "LeNet",
        "sampling_mode": "subsampling",
        "clipping_method": "IC",
        "clipping_mode": "layerwise",
        "constant_step_size": True,
        "DGN": False,
        "AN": False,
        "Optimizer": "SGD",
    }
    train_accuracy, test_accuracy, epochs = get_data_from_settings(line_2_setting_info)
    test_acc_IC_LeNet_layerwise_css.append(test_accuracy)
    """Line 3"""
    line_3_setting_info = {
        "data_folder": "data_neurips",
        "setting_path_base": "settings_clipping_exp_cifar10_dpsgd_opacus",
        "setting_index": setting,
        "model_name": "LeNet",
        "sampling_mode": "subsampling",
        "clipping_method": "BC",
        "clipping_mode": "layerwise",
        "constant_step_size": False,
        "DGN": False,
        "AN": False,
        "Optimizer": "SGD",
    }
    train_accuracy, test_accuracy, epochs = get_data_from_settings(line_3_setting_info)
    test_acc_BC_LeNet_layerwise.append(test_accuracy)
    """Line 4"""
    line_4_setting_info = {
        "data_folder": "data_neurips",
        "setting_path_base": "settings_clipping_exp_cifar10_dpsgd_opacus",
        "setting_index": setting,
        "model_name": "LeNet",
        "sampling_mode": "subsampling",
        "clipping_method": "BC",
        "clipping_mode": "layerwise",
        "constant_step_size": True,
        "DGN": False,
        "AN": False,
        "Optimizer": "SGD",
    }
    train_accuracy, test_accuracy, epochs = get_data_from_settings(line_4_setting_info)
    test_acc_BC_LeNet_layerwise_css.append(test_accuracy)
    """Line 5"""
    line_5_setting_info = {
        "data_folder": "data_neurips",
        "setting_path_base": "settings_clipping_exp_cifar10_dpsgd_opacus",
        "setting_index": setting,
        "model_name": "nor_LeNet",
        "sampling_mode": "subsampling",
        "clipping_method": "BC",
        "clipping_mode": "layerwise",
        "constant_step_size": True,
        "DGN": False,
        "AN": False,
        "Optimizer": "SGD",
    }
    train_accuracy, test_accuracy, epochs = get_data_from_settings(line_5_setting_info)
    test_acc_BC_LeNet_layerwise_BN.append(test_accuracy)
    """Line 6"""
    line_6_setting_info = {
        "data_folder": "data_neurips",
        "setting_path_base": "settings_clipping_exp_cifar10_dpsgd_opacus",
        "setting_index": setting,
        "model_name": "nor_LeNet",
        "sampling_mode": "subsampling",
        "clipping_method": "BC",
        "clipping_mode": "layerwise",
        "constant_step_size": True,
        "DGN": False,
        "AN": False,
        "Optimizer": "SGD",
    }
    train_accuracy, test_accuracy, epochs = get_data_from_settings(line_6_setting_info)
    test_acc_BC_LeNet_layerwise_css_BN.append(test_accuracy)
    """Line 7"""
    line_7_setting_info = {
        "data_folder": "data_neurips",
        "setting_path_base": "settings_clipping_exp_cifar10_dpsgd_opacus",
        "setting_index": setting,
        "model_name": "nor_LeNet",
        "sampling_mode": "subsampling",
        "clipping_method": None,
        "clipping_mode": "layerwise",
        "constant_step_size": True,
        "DGN": False,
        "AN": False,
        "Optimizer": "SGD",
    }
    train_accuracy, test_accuracy, epochs = get_data_from_settings(line_7_setting_info)
    test_acc_baseline.append(test_accuracy)
    """"""
    epoch_index = [i for i in range(1, epochs+1)]
    epoch_index = np.array(epoch_index)
    """"""

    """ DRAW PLOT"""
    """Line1"""
    """IC + layerwise"""
    test_accuracy = np.array(np.mean(test_acc_IC_LeNet_layerwise, axis=0))

    # 500 represents number of points to make between T.min and T.max
    xnew = np.linspace(epoch_index.min(), epoch_index.max(), 500)
    X_Y_Spline = make_interp_spline(epoch_index, test_accuracy)
    test_accuracy_smooth = X_Y_Spline(xnew)
    #
    label = "%s , %s" % (line_1_setting_info["clipping_method"], line_1_setting_info["clipping_mode"])
    # plt.plot(xnew,test_accuracy_smooth,label=label)
    plt.plot(epoch_index, test_accuracy,label=label)

    """Line2"""
    """IC + layerwise + css"""
    test_accuracy = np.array(np.mean(test_acc_IC_LeNet_layerwise_css, axis=0))

    # 500 represents number of points to make between T.min and T.max
    xnew = np.linspace(epoch_index.min(), epoch_index.max(), 500)
    X_Y_Spline = make_interp_spline(epoch_index, test_accuracy)
    test_accuracy_smooth = X_Y_Spline(xnew)
    #
    label = "%s , %s, css" % (line_2_setting_info["clipping_method"], line_2_setting_info["clipping_mode"])
    # plt.plot(xnew,test_accuracy_smooth,label=label)
    plt.plot(epoch_index, test_accuracy,label=label)

    """Line3"""
    """BC + layerwise """
    test_accuracy = np.array(np.mean(test_acc_BC_LeNet_layerwise, axis=0))

    # 500 represents number of points to make between T.min and T.max
    xnew = np.linspace(epoch_index.min(), epoch_index.max(), 500)
    X_Y_Spline = make_interp_spline(epoch_index, test_accuracy)
    test_accuracy_smooth = X_Y_Spline(xnew)
    #
    label = "%s , %s" % (line_3_setting_info["clipping_method"], line_3_setting_info["clipping_mode"])
    # plt.plot(xnew,test_accuracy_smooth,label=label)
    plt.plot(epoch_index, test_accuracy,label=label)

    """Line4"""
    """BC + layerwise +css """
    test_accuracy = np.array(np.mean(test_acc_BC_LeNet_layerwise_css, axis=0))

    # 500 represents number of points to make between T.min and T.max
    xnew = np.linspace(epoch_index.min(), epoch_index.max(), 500)
    X_Y_Spline = make_interp_spline(epoch_index, test_accuracy)
    test_accuracy_smooth = X_Y_Spline(xnew)
    #
    label = "%s , %s, css" % (line_4_setting_info["clipping_method"], line_4_setting_info["clipping_mode"])
    # plt.plot(xnew,test_accuracy_smooth,label=label)
    plt.plot(epoch_index, test_accuracy,label=label)

    """Line5"""
    """BC + layerwise + BN """
    test_accuracy = np.array(np.mean(test_acc_BC_LeNet_layerwise_BN, axis=0))

    # 500 represents number of points to make between T.min and T.max
    xnew = np.linspace(epoch_index.min(), epoch_index.max(), 500)
    X_Y_Spline = make_interp_spline(epoch_index, test_accuracy)
    test_accuracy_smooth = X_Y_Spline(xnew)
    #
    label = "%s , %s, BN" % (line_5_setting_info["clipping_method"], line_5_setting_info["clipping_mode"])
    # plt.plot(xnew,test_accuracy_smooth,label=label)
    plt.plot(epoch_index, test_accuracy,label=label)

    """Line 6"""
    """BC + layerwise + BN + css """
    test_accuracy = np.array(np.mean(test_acc_BC_LeNet_layerwise_css_BN, axis=0))

    # 500 represents number of points to make between T.min and T.max
    xnew = np.linspace(epoch_index.min(), epoch_index.max(), 500)
    X_Y_Spline = make_interp_spline(epoch_index, test_accuracy)
    test_accuracy_smooth = X_Y_Spline(xnew)
    #
    label = "%s , %s, BN, css" % (line_6_setting_info["clipping_method"], line_6_setting_info["clipping_mode"])
    # plt.plot(xnew,test_accuracy_smooth,label=label)
    plt.plot(epoch_index, test_accuracy,label=label)

    """Line 7"""
    """Baseline """
    test_accuracy = np.array(np.mean(test_acc_baseline, axis=0))

    # 500 represents number of points to make between T.min and T.max
    xnew = np.linspace(epoch_index.min(), epoch_index.max(), 500)
    X_Y_Spline = make_interp_spline(epoch_index, test_accuracy)
    test_accuracy_smooth = X_Y_Spline(xnew)
    #
    label = "baseline (with BN)"
    # plt.plot(xnew,test_accuracy_smooth,label=label)
    plt.plot(epoch_index, test_accuracy,label=label)



    """----------------------------------------------"""
    plt.title("Testing accuracy on MNIST dataset using LeNet model with/without BatchNorm layer")
    plt.xlabel("epoch")
    plt.ylabel("Testing accuracy")
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


