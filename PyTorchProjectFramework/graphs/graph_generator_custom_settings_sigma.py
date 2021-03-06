import matplotlib.pyplot  as plt
import numpy

import os
import json

def get_data_from_settings(setting_path,setting_name,model_name,experiment,IC,BC):
    data_path  = "./data/" + setting_path + '/'  + model_name + '/' + experiment + '/' + setting_name +".json" #SGD
    if(BC):
        data_path  = "./data/" + setting_path + '/'  + model_name + '/' + experiment + '/BC/' + setting_name +".json"
    if(IC):
        data_path  = "./data/" + setting_path + '/'  + model_name + '/' + experiment + '/IC/' + setting_name +".json"
    with open(data_path, "r") as data_file:
        data = json.load(data_file)
        DPSGD_train_accuracy = data["train_accuracy"]
        DPSGD_test_accuracy = data["test_accuracy"]
        DPSGD_epochs = len(DPSGD_train_accuracy)
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
    C = 10
    lr = 0.1
    draw_IC_case = False
    draw_BC_case = True
    label = "BC sigma = %f, s = %f" if draw_BC_case else "IC sigma = %f, s = %f"
    setup_plot('epoch' , 'accuracy',lr ,C)
    """SGD DATA 256"""
    model_name = "convnet"
    setting_path = "settings_clipping_exp_cifar10_dpsgd_large_C"
    s = 256
    setting_name = "setting_24"
    experiment = "SGD"

    train_accuracy, test_accuracy, epochs = get_data_from_settings(
        setting_path,setting_name,
        model_name,experiment,
        False,False)
    epoch_index = [i for i in range(1, epochs+1)]
    plt_draw(epoch_index, train_accuracy,test_accuracy,"SGD s = 256",None,None)
    """SGD DATA 512"""
    model_name = "convnet"
    setting_path = "settings_clipping_exp_cifar10_dpsgd_large_C"
    s = 512
    setting_name = "setting_25"
    experiment = "SGD"

    train_accuracy, test_accuracy, epochs = get_data_from_settings(
        setting_path,setting_name,
        model_name,experiment,
        False,False)
    epoch_index = [i for i in range(1, epochs+1)]
    plt_draw(epoch_index, train_accuracy,test_accuracy,"SGD, s = 512",None,None)
    """SETTING 1 DATA"""
    model_name = "convnet"
    setting_path = "settings_clipping_exp_cifar10_dpsgd_large_C"
    
    s = 256
    sigma = 2
    setting_name = "setting_24"
    experiment = "SGD"

    train_accuracy, test_accuracy, epochs = get_data_from_settings(
        setting_path,setting_name,
        model_name,experiment,
        draw_IC_case,draw_BC_case)
    epoch_index = [i for i in range(1, epochs+1)]
    plt_draw(epoch_index, train_accuracy,test_accuracy,label,sigma,s)

    """SETTING 2 DATA"""
    model_name = "convnet"
    setting_path = "settings_clipping_exp_cifar10_dpsgd_large_C"
    
    s = 512
    sigma = 2
    setting_name = "setting_25"
    experiment = "SGD"

    train_accuracy, test_accuracy, epochs = get_data_from_settings(
        setting_path,setting_name,
        model_name,experiment,
        draw_IC_case,draw_BC_case)
    epoch_index = [i for i in range(1, epochs+1)]
    plt_draw(epoch_index, train_accuracy,test_accuracy,label,sigma,s)
    plt.legend()

    """SETTING 3 DATA"""
    model_name = "convnet"
    setting_path = "settings_clipping_exp_cifar10_dpsgd_large_C_sigma_4"
    
    s = 256
    sigma = 4
    setting_name = "setting_24"
    experiment = "SGD"

    train_accuracy, test_accuracy, epochs = get_data_from_settings(
        setting_path,setting_name,
        model_name,experiment,
        draw_IC_case,draw_BC_case)
    epoch_index = [i for i in range(1, epochs+1)]
    plt_draw(epoch_index, train_accuracy,test_accuracy,label,sigma,s)
    plt.legend()

    """SETTING 4 DATA"""
    model_name = "convnet"
    setting_path = "settings_clipping_exp_cifar10_dpsgd_large_C_sigma_4"

    s = 512
    sigma = 4
    setting_name = "setting_25"
    experiment = "SGD"

    train_accuracy, test_accuracy, epochs = get_data_from_settings(
        setting_path,setting_name,
        model_name,experiment,
        draw_IC_case,draw_BC_case)
    epoch_index = [i for i in range(1, epochs+1)]
    plt_draw(epoch_index, train_accuracy,test_accuracy,label,sigma,s)
    plt.legend()

    """SETTING 5 DATA"""
    model_name = "convnet"
    setting_path = "settings_clipping_exp_cifar10_dpsgd_large_C_sigma_8"

    s = 256
    sigma = 8
    setting_name = "setting_24"
    experiment = "SGD"

    train_accuracy, test_accuracy, epochs = get_data_from_settings(
        setting_path,setting_name,
        model_name,experiment,
        draw_IC_case,draw_BC_case)
    epoch_index = [i for i in range(1, epochs+1)]
    plt_draw(epoch_index, train_accuracy,test_accuracy,label,sigma,s)

    """SETTING 6 DATA"""
    model_name = "convnet"
    setting_path = "settings_clipping_exp_cifar10_dpsgd_large_C_sigma_8"

    s = 512
    sigma = 8
    setting_name = "setting_25"
    experiment = "SGD"

    train_accuracy, test_accuracy, epochs = get_data_from_settings(
        setting_path,setting_name,
        model_name,experiment,
        draw_IC_case,draw_BC_case)
    epoch_index = [i for i in range(1, epochs+1)]
    plt_draw(epoch_index, train_accuracy,test_accuracy,label,sigma,s)

    # """SETTING 7 DATA"""
    # model_name = "convnet"
    # setting_path = "settings_clipping_exp_cifar10_dpsgd_large_C"
    #
    # s = 256
    # C = 20.0
    # setting_name = "setting_29"
    # experiment = "SGD"
    #
    # train_accuracy, test_accuracy, epochs = get_data_from_settings(
    #     setting_path,setting_name,
    #     model_name,experiment,
    #     draw_IC_case,draw_BC_case)
    # epoch_index = [i for i in range(1, epochs+1)]
    # plt_draw(epoch_index, train_accuracy,test_accuracy,label,sigma,s)
    plt.legend()


    graph_path = "./graph/C_custom_compare"
    # Check whether the specified path exists or not
    isExist = os.path.exists(graph_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(graph_path)
        print("The new directory is created: %s" % graph_path)

        # s = s*2
    file_name = '/dpsgd_sigma_comparing_lr_' + str(lr) + '_C_' + str(C)
    if (draw_IC_case):
        file_name = '/IC_dpsgd_sigma_comparing_lr_' + str(lr) + '_C_' + str(C)
    if (draw_BC_case):
        file_name = '/BC_dpsgd_sigma_comparing_lr_' + str(lr) + '_C_' + str(C)
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


