import json

settings = ["settings_clipping_exp_cifar10_dpsgd",
            "settings_clipping_exp_cifar10_dpsgd_sigma_4",
            "settings_clipping_exp_cifar10_dpsgd_sigma_8",
            "settings_clipping_exp_cifar10_dpsgd_new",
            "settings_clipping_exp_cifar10_dpsgd_new_sigma_4",
            "settings_clipping_exp_cifar10_dpsgd_new_sigma_8",
            "settings_clipping_exp_cifar10_dpsgd_large_C",
            "settings_clipping_exp_cifar10_dpsgd_large_C_sigma_4",
            "settings_clipping_exp_cifar10_dpsgd_large_C_sigma_8",
            "settings_clipping_exp_cifar10_dpsgd_opacus",
            "settings_clipping_exp_cifar10_dpsgd_opacus_sigma_4",
            "settings_clipping_exp_cifar10_dpsgd_opacus_sigma_8",
            "settings_clipping_exp_cifar10_dpsgd_opacus_sigma_p5",
            "settings_clipping_exp_cifar10_dpsgd_opacus_sigma_1p5",]
settings = ["settings_best_settings_lost_func_grid_search_1"]
settings = ["settings_clipping_exp_cifar10_dpsgd_opacus"]
settings = ["settings_convnet"]

# settings = ["settings_lost_func_grid_search_sigma2_6"]
# .json
# settings = ["settings_clipping_exp_cifar10_dpsgd_opacus_test"]

# base_sigma = 0.1
# C = 1.2
# base_sigma = 0.5
base_sigma = 0.05

# C = 0.01
# C = 0.08
# C = 0.64

# C = 3.84
# C = 23.04
# C = 138.24
C = 0.1
enable_loss_multi = False
base_loss_multi = 1
# base_loss_multi = pow(2,10)
"""
Sampler mode
"""
data_processing = "subsampling"
# data_processing = "shuffling"

"""
Clipping mode
"""
is_batch_clipping = True
is_individual_clipping = False
is_classical_BC = False

"""
Stepsize mode
"""
is_constant_step_size = False
count = 0
for setting_file in settings:
# setting_file = settings[0]
    print(setting_file)
    f = open(setting_file +".json")
    data = json.load(f)
    f.close()
    output_file = setting_file
    """Update elements"""
    for (k, v) in data.items():
                                # count = 0 1 2 3 4 5 6 7 8 9
        C_multi = (count % 5 ) + 1      # 1 2 3 4 5 1 2 3 4 5
        sigma_multi = (count // 5) + 1  # 1 1 1 1 1 2 2 2 2 2
        data[k]['batch_size'] = 1028
        data[k]['max_grad_norm'] = round(C * pow(2,C_multi),2)
        data[k]['noise_multiplier'] = round(base_sigma * pow(2,sigma_multi),2)
        data[k]['learning_rate'] = 0.001
        if(enable_loss_multi):
            data[k]['loss_multi'] = base_loss_multi * sigma_multi
        if(is_constant_step_size):
            data[k]['gamma'] = 1
        else:
            data[k]['gamma'] = 0.9
        
        # data[k]['learning_rate'] = 0.025
        data[k]['data_sampling'] = data_processing
        print("Key: " + k)
        print("Value: " + str(v))
        count = count + 1
    """Output files"""
    # if(is_batch_clipping):
    #     output_file = output_file + "_" + data_processing +"_BC"
    # elif(is_individual_clipping):
    #     output_file = output_file + "_" + data_processing +"_IC"
    # elif(is_classical_BC):
    #     output_file = output_file + "_" + data_processing +"_classical"
    # else:
    #     output_file = output_file + "_" + data_processing
    if(is_constant_step_size):
        output_file = output_file + "_css"
    print("output_file:", output_file + ".json")
    with open(output_file + "_1028_lr_low.json", "w") as data_file:
        json.dump(data, data_file,indent=2)