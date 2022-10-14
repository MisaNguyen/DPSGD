import json

setting_file = "settings_clipping_exp_cifar10_dpsgd_new_sigma_8.json"
f = open(setting_file)
data = json.load(f)
f.close()
"""Update elements"""
for (k, v) in data.items():
    data[k]['batch_size'] = data[k]['batch_size'] *4
    data[k]['microbatch_size'] = data[k]['microbatch_size'] *4
    print("Key: " + k)
    print("Value: " + str(v))
"""Output files"""
with open(setting_file, "w") as data_file:
    json.dump(data, data_file,indent=2)