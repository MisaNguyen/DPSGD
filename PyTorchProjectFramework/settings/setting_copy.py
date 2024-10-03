import json
import shutil
# settings = ["settings_clipping_exp_cifar10_dpsgd",
#             "settings_clipping_exp_cifar10_dpsgd_sigma_4",
#             "settings_clipping_exp_cifar10_dpsgd_sigma_8",
#             "settings_clipping_exp_cifar10_dpsgd_new",
#             "settings_clipping_exp_cifar10_dpsgd_new_sigma_4",
#             "settings_clipping_exp_cifar10_dpsgd_new_sigma_8",
#             "settings_clipping_exp_cifar10_dpsgd_large_C",
#             "settings_clipping_exp_cifar10_dpsgd_large_C_sigma_4",
#             "settings_clipping_exp_cifar10_dpsgd_large_C_sigma_8",
#             "settings_clipping_exp_cifar10_dpsgd_opacus",
#             "settings_clipping_exp_cifar10_dpsgd_opacus_sigma_4",
#             "settings_clipping_exp_cifar10_dpsgd_opacus_sigma_8",
#             "settings_clipping_exp_cifar10_dpsgd_opacus_sigma_p5",
#             "settings_clipping_exp_cifar10_dpsgd_opacus_sigma_1p5",]
# settings = ["settings_best_settings_lost_func_grid_search_6"]
settings = [
"settings_lost_func_grid_search_sigma1_1",
"settings_lost_func_grid_search_sigma1_2",
"settings_lost_func_grid_search_sigma1_3",
"settings_lost_func_grid_search_sigma1_4",
"settings_lost_func_grid_search_sigma1_5",
"settings_lost_func_grid_search_sigma1_6",
"settings_lost_func_grid_search_sigma1_7",
"settings_lost_func_grid_search_sigma1_8",
"settings_lost_func_grid_search_sigma1_9",
"settings_lost_func_grid_search_sigma1_10",
"settings_lost_func_grid_search_sigma1_11",
"settings_lost_func_grid_search_sigma1_12"
]
settings_after = [
    "settings_lost_func_grid_search_sigma2_1",
    "settings_lost_func_grid_search_sigma2_2",
    "settings_lost_func_grid_search_sigma2_3",
    "settings_lost_func_grid_search_sigma2_4",
    "settings_lost_func_grid_search_sigma2_5",
    "settings_lost_func_grid_search_sigma2_6",
    "settings_lost_func_grid_search_sigma2_7",
    "settings_lost_func_grid_search_sigma2_8",
    "settings_lost_func_grid_search_sigma2_9",
    "settings_lost_func_grid_search_sigma2_10",
    "settings_lost_func_grid_search_sigma2_11",
    "settings_lost_func_grid_search_sigma2_12"
]

for setting_idx in range(len(settings)):
    src = settings[setting_idx] + ".json"
    dst = settings_after[setting_idx] + ".json"
    shutil.copy2(src, dst)