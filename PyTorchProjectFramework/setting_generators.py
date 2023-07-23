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
settings = ["settings_best_settings_lost_func_grid_search_6"]
# settings = ["settings_clipping_exp_cifar10_dpsgd_opacus_test"]

# base_sigma = 0.1
# C = 1.2
base_sigma = 0.01875
C = 0.64
base_loss_multi = pow(2,10)
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
is_constant_step_size = True
count = 0
for setting_file in settings:
# setting_file = settings[0]
    print(setting_file)
    f = open(setting_file +".json")
    data = json.load(f)
    f.close()
    """Update elements"""
    for (k, v) in data.items():
        data[k]['batch_size'] = 64
        data[k]['loss_multi'] = base_loss_multi * pow(2,count%10-5)
        if(is_batch_clipping):
            data[k]['microbatch_size'] = data[k]['batch_size']
        elif(is_individual_clipping):
            data[k]['microbatch_size'] = 1
        elif(is_classical_BC):
            data[k]['microbatch_size'] = 64

        if(is_constant_step_size):
            data[k]['gamma'] = 1
        data[k]['max_grad_norm'] = C * ((count+10)//10)
        data[k]['noise_multiplier'] = base_sigma
        # data[k]['learning_rate'] = 0.025
        data[k]['data_sampling'] = data_processing
        print("Key: " + k)
        print("Value: " + str(v))
        count = count + 1
    """Output files"""
    if(is_batch_clipping):
        output_file = setting_file + "_" + data_processing +"_BC"
    elif(is_individual_clipping):
        output_file = setting_file + "_" + data_processing +"_IC"
    elif(is_classical_BC):
        output_file = setting_file + "_" + data_processing +"_classical"
    else:
        output_file = setting_file + "_" + data_processing
    if(is_constant_step_size):
        output_file = output_file +"_css"
    print("output_file:", output_file + ".json")
    with open(output_file + ".json", "w") as data_file:
        json.dump(data, data_file,indent=2)