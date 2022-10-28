import json

settings = ["settings_clipping_exp_cifar10_dpsgd",
            "settings_clipping_exp_cifar10_dpsgd_sigma_4",
            "settings_clipping_exp_cifar10_dpsgd_sigma_8",
            "settings_clipping_exp_cifar10_dpsgd_new",
            "settings_clipping_exp_cifar10_dpsgd_new_sigma_4",
            "settings_clipping_exp_cifar10_dpsgd_new_sigma_8",
            "settings_clipping_exp_cifar10_dpsgd_large_C",
            "settings_clipping_exp_cifar10_dpsgd_large_C_sigma_4",
            "settings_clipping_exp_cifar10_dpsgd_large_C_sigma_8",]
for setting_file in settings:
    setting_file = setting_file +".json"
# setting_file = settings[0]
    f = open(setting_file )
    data = json.load(f)
    f.close()
    """Update elements"""
    for (k, v) in data.items():
        # data[k]['batch_size'] = data[k]['batch_size'] *4
        # data[k]['microbatch_size'] = data[k]['microbatch_size'] *4
        # data[k]['learning_rate'] = 0.025
        data[k]['is_individual_clipping'] = False
        print("Key: " + k)
        print("Value: " + str(v))
    """Output files"""
    with open(setting_file, "w") as data_file:
        json.dump(data, data_file,indent=2)