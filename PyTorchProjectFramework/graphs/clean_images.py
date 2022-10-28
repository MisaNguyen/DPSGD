import os, shutil
settings = ["settings_clipping_exp_cifar10_dpsgd",
            "settings_clipping_exp_cifar10_dpsgd_sigma_4",
            "settings_clipping_exp_cifar10_dpsgd_sigma_8",
            "settings_clipping_exp_cifar10_dpsgd_new",
            "settings_clipping_exp_cifar10_dpsgd_new_sigma_4",
            "settings_clipping_exp_cifar10_dpsgd_new_sigma_8",
            "settings_clipping_exp_cifar10_dpsgd_large_C",
            "settings_clipping_exp_cifar10_dpsgd_large_C_sigma_4",
            "settings_clipping_exp_cifar10_dpsgd_large_C_sigma_8"]
for setting in settings:
    # graph_path = "./graph/" + setting + '/clipping'
    graph_path = "./graph/" + setting + '/C_compare'
    for filename in os.listdir(graph_path):
        file_path = os.path.join(graph_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
