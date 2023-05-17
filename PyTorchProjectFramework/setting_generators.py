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
settings = ["settings_best_settings"]
base_sigma = 0.01875
C = 0.095
data_processing = "subsampling"
# data_processing = "shuffling"

is_batch_clipping = True
is_individual_clipping = False
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
        if(is_batch_clipping):
            data[k]['microbatch_size'] = data[k]['batch_size']
        elif(is_individual_clipping):
            data[k]['microbatch_size'] = 1
        data[k]['max_grad_norm'] = C
        data[k]['noise_multiplier'] = base_sigma
        # data[k]['learning_rate'] = 0.025
        data[k]['data_sampling'] = data_processing
        print("Key: " + k)
        print("Value: " + str(v))
        count = count + 1
    """Output files"""
    if(is_batch_clipping):
        output_file = setting_file + "_" + data_processing +"_BC.json"
    elif(is_individual_clipping):
        output_file = setting_file + "_" + data_processing +"_IC.json"
    else:
        output_file = setting_file + "_" + data_processing +".json"
    with open(output_file, "w") as data_file:
        json.dump(data, data_file,indent=2)