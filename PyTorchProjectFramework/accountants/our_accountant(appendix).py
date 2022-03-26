import pandas as pd
import numpy as np
import json
# import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt


from gdp_accountant_tensor import compute_eps_poisson,compute_mu_poisson, compute_eps_uniform, compute_mu_uniform
import os
# Compute the value of gamma
def Compute_gamma(alpha_bar,sigma):
    C_0 = 2/(1-alpha_bar) # 2/(1-alpha)
    C_1 = pow(2,4)*alpha_bar/(1-alpha_bar) # 2^4*alpha/(1-alpha)
    C_2 = sigma/pow(1-np.sqrt(alpha_bar),2) # sigma/(1-sqrt(alpha))^2
    C_3 = (1/(sigma*(1-alpha_bar) - 2 * np.exp(1)* np.sqrt(alpha_bar)))*np.exp(3)/sigma # 1/(sigma*(1-alpha))
    C_4 = np.exp(3/pow(sigma,2))
    C_5 = 2/(1-alpha_bar)
    return C_0 + C_1*(C_2+C_3)*C_4 + C_5

# Compute upper bound on CONSTANT sample size
def Compute_upperbound_s(N_c,K,eps,sigma,theta):
    alpha_bar = Compute_alpha_bar(eps,N_c,K)
    gamma = Compute_gamma(alpha_bar,sigma)
    return (eps/(gamma*pow(theta,2)*K))*pow(N_c,2)

def Compute_h(x):
    C_1 = np.sqrt(1+pow(np.exp(1)/x,2))
    C_2= pow(C_1-np.exp(1)/x,2)
    return C_2

def Compute_g(x):
    return min(1/(np.exp(1)*x),Compute_h(x))

def Compute_alpha_bar(eps,N_c,K,gamma):
    return eps*N_c/(gamma*K) # N_c/K = 1/k

def find_min_gamma(eps_left,eps_right,N_c,K):
    alpha_bar_left = Compute_alpha_bar(eps_left,N_c,K)
    if (alpha_bar_left >= 1):
        return np.inf
    sigma_left = np.sqrt(2*np.log(1/delta)/eps_left)
    gamma_left = Compute_gamma(alpha_bar_left,sigma_left)

    alpha_bar_right = Compute_alpha_bar(eps_right,N_c,K)
    if (alpha_bar_right >= 1):
        return np.inf
    sigma_right = np.sqrt(2*np.log(1/delta)/eps_right)
    gamma_right = Compute_gamma(alpha_bar_right,sigma_right)

    if(abs(gamma_right - gamma_left) < 0.01):
        if(gamma_right > gamma_left):
            return eps_left
        else:
            return eps_right
    eps_mean = (eps_left + eps_right)/2
    alpha_bar_mean = Compute_alpha_bar(eps_mean,N_c,K)
    if (alpha_bar_mean >= 1):
        return np.inf
    sigma_mean = np.sqrt(2*np.log(1/delta)/eps_mean)
    gamma_mean = Compute_gamma(alpha_bar_mean,sigma_mean)
    # Return the eps which produces smaller gamma if gamma_mean is greater than gamma_left or gamma_right
    if(gamma_mean > min(gamma_left,gamma_right)):
        if(gamma_right > gamma_left):
            return eps_left
        else:
            return eps_right
    else:
        return min(find_min_gamma(eps_left,eps_mean,N_c,K),find_min_gamma(eps_mean,eps_right,N_c,K))
# Output T
# s_i <= np.sqrt(eps/(gamma*T))*N_c
# sigma >= np.sqrt(2*np.log(1/delta)/eps) => (eps,delta) -DP
def DP_calculator_CSS(N_c,K,delta,s,theta = 1):
    # theta = 1 # because s_mean = s_max in constant sample size case
    # eps = None
    # sigma = None
    # gamma_start = 2
    gamma = 0
    new_gamma = 2
    # Compute until gamma is converge
    while(new_gamma - gamma >= 0.0001*gamma):
        gamma = new_gamma
        eps = gamma*pow(theta,2)*K/pow(N_c,2)*s
        # print("eps = %f" % eps)
        sigma = np.sqrt(2*(eps + np.log(1/delta))/eps)

        alpha_bar = Compute_alpha_bar(eps,N_c,K,gamma)
        if (alpha_bar >= 1):
            break
        new_gamma = Compute_gamma(alpha_bar,sigma)
        # print(Compute_upperbound_s(N_c,K,eps,sigma,theta))
        # input()
        # print(gamma_start)
        # print(eps)
        # print(sigma)
        # print(new_gamma)
        # input()
    gamma = new_gamma
    # print(eps)
    # print(sigma)
    # print(alpha_bar)
    # print(gamma)
    # print(new_gamma)
    # print(Compute_gamma(alpha_bar,sigma))
    # sigma_min = np.sqrt(2*np.log(1/delta)/eps_min)
    # if(sigma < sigma_min):
    #     print("Error: Sigma must greater than or equal to", sigma_min)
    #     return None
    # eps_min = 2*np.log(1/delta)/pow(sigma,2)
    # Binary search for eps such that gamma = gamma_min

    # alpha_bar = Compute_alpha_bar(eps_min,N_c,K)
    # # print(alpha_bar)
    # if (alpha_bar >= 1):
    #     print("Error: eps*N_c  must be less than 2*s*T")
    #     return None, None
    # sigma_lowerbound = 2*np.exp(1)*np.sqrt(alpha_bar)/(1-alpha_bar)
    # if(sigma <= sigma_lowerbound):
    #     print("Error: Sigma must greater than ", sigma_lowerbound)
    #     return None
    # min gamma/alpha_bar^2 => alpha_bar = 0.15
    # alpha_bar = 0.15
    # gamma_left = Compute_gamma(alpha_bar,sigma)
    # gamma_left = Compute_gamma(alpha_bar,sigma_min)
    # i = 1
    # rho = 2
    # gamma_right = 2
    # # Find gamma_right
    # while(gamma_left > gamma_right):
    #     eps_max = pow(rho,i)* eps_min
    #     sigma_max = np.sqrt(2*np.log(1/delta)/eps_max)
    #     alpha_bar = Compute_alpha_bar(eps_max,N_c,K)
    #     gamma_right = Compute_gamma(alpha_bar,sigma_max)
    #     i = i + 1
    # print("gamma_left=",gamma_left)
    # print("eps_max=",eps_min)
    # print("eps_max=",eps_max)
    # print("gamma_right=",gamma_right)
    #--------------------------------------------------------------
    # eps_arr = [eps_min + i/1000 for i in range(100,1000)]
    # alpha_bar_arr = []
    # gamma_arr = []
    # for eps in eps_arr:
    #     # sigma = np.sqrt(2*np.log(1/delta)/eps)
    #     # print(item)

    #     # print(sigma)
    #     sigma = np.sqrt(2*np.log(1/delta)/eps)
    #     alpha_bar = Compute_alpha_bar(eps,N_c,K)
    #     alpha_bar_arr.append(alpha_bar) # 0.01 -> 0.5
    #     gamma_arr.append(Compute_gamma(alpha_bar,sigma))
    # fig, axes = plt.subplots(nrows=2, ncols=2)

    # axes[0][0].plot(eps_arr, alpha_bar_arr)
    # axes[0][0].set_title('N_c = 60000, delta = 1/N_c')
    # axes[0][0].set_xlabel('eps')
    # axes[0][0].set_ylabel('alpha_bar')

    # axes[0][1].plot(eps_arr, gamma_arr)
    # axes[0][1].set_title('N_c = 60000, delta = 1/N_c')
    # axes[0][1].set_xlabel('eps')
    # axes[0][1].set_ylabel('gamma')
    # plt.show()
    # input()
    #--------------------------------------------------------------
    # find eps in [eps_min,eps_max] such that gamma = gamma_min
    # eps = find_min_gamma(eps_min,eps_max,N_c,K)
    # sigma = np.sqrt(2*np.log(1/delta)/eps)
    # alpha_bar = Compute_alpha_bar(eps,N_c,K)
    # gamma = Compute_gamma(alpha_bar,sigma)
    # print("eps_min =",eps_min)
    # print("eps =",eps)
    # print("eps_max =",eps_max)
    # print("gamma=",gamma)



    # Checking condition:
    # Condition 1:
    # s_ub = Compute_upperbound_s(N_c,K,eps,sigma,theta)
    RHS_condition1 = Compute_g(np.sqrt(2*np.log(1/delta)/eps))/theta * N_c
    # print("1",RHS_condition1)

    # print("2",s_ub)
    if (s > RHS_condition1):
        print("ERROR : s must be smaller than or equal to" , RHS_condition1)
        return None, None

    # Condition 2:
    RHS_condition2 = gamma*Compute_h(sigma)*K/N_c
    if(eps > RHS_condition2):
        print("ERROR : eps must be smaller than or equal to" , RHS_condition2)
        return None, None
    # T_min = eps*pow(np.exp(1)*sigma,2)/gamma
    # T_max = eps*pow(N_c,2)/(gamma*pow(s,2))
    # T = K/s

    # print("T_min=",T_min)
    # print("T=",T)
    # print("T_max=",T_max)
    # input()
    # if(T < T_min or T > T_max):
    #     print("T=", T)
    #     print("Error: T must be in range: " + "[" + str(T_min) + "," + str(T_max) + "]")
    #     print("Solution: decrease s")
    #     return None
    # s = np.sqrt(eps/(gamma*T))*N_c

    return eps,sigma


def compute_lower_T(gamma,eps, K, N, theta=1):
    k = K/N
    return (gamma * theta * theta * k * k)/eps


def compute_eps_from_sigma_and_delta(sigma,delta,K,N_c,s,theta = 1):
    # sigma >= sqrt(2(eps+ln(1/delta))/eps)
    # sigma^2*eps >= 2(eps+ln(1/delta))
    # sigma^2* eps - 2 eps > = ln(1/delta)
    # eps => 2 *(ln(1/delta))/ (sigma^2 -2) (this means sigma >= sqrt(2))
    eps_min = 2 *np.log(1/delta)/ (sigma*sigma -2)
    alpha_bar = Compute_alpha_bar(eps_min,N_c,K,gamma=2) # Start with gamma=2
    # print(alpha_bar)
    gamma = 0
    new_gamma = Compute_gamma(alpha_bar,sigma)
    # print(new_gamma)
    # Compute until gamma is converge
    # print(eps_min)
    while(new_gamma - gamma >= 0.0001*gamma):
        gamma = new_gamma

        eps = gamma * pow(theta,2) * K / pow(N_c,2)*s
        # print(eps_min)
        # input(eps)
        # print("eps = %f" % eps)
        # sigma = np.sqrt(2*(eps + np.log(1/delta))/eps)

        alpha_bar = Compute_alpha_bar(eps,N_c,K,gamma)
        if (alpha_bar >= 1):
            break
        new_gamma = Compute_gamma(alpha_bar,sigma)
    # if (alpha_bar >= 1):

        # break
    # new_gamma = Compute_gamma(alpha_bar,sigma)
    RHS_condition1 = Compute_g(np.sqrt(2*np.log(1/delta)/eps))/theta * N_c
    # print("1",RHS_condition1)

    # print("2",s_ub)
    if (s > RHS_condition1):
        print("ERROR : s must be smaller than or equal to" , RHS_condition1)
        return None, None

    # Condition 2:
    RHS_condition2 = gamma*Compute_h(sigma)*K/N_c
    if(eps > RHS_condition2):
        print("ERROR : eps must be smaller than or equal to" , RHS_condition2)
        return None, None
    # Condition 3:
    RHS_condition3 = np.sqrt(2*(eps + np.log(1/delta))/eps)
    if(sigma < RHS_condition3):
        print("ERROR : sigma must be larger than or equal to" , RHS_condition3)
        return None
    return eps

"""
Setting sample:
    "sampling_batch": 64,
    "num_examples": 60000,
    "epochs": 50,
    "learning_rate": 0.001,
    "noise_multiplier": 1.0,
    "max_grad_norm": 0.001,
"""
if __name__ == "__main__":
    # N_c,K,delta,s = setting_3()
    setting_file_name = "settings_sample_size_exp"
    # settings = ["setting_1","setting_2","setting_3","setting_4","setting_5","setting_6"]
    settings = ["setting_1"]
    with open("./%s.json" % setting_file_name, "r") as json_file:
        json_data = json.load(json_file)

    json_output_test = json_data
    for setting in settings:
        setting_data = json_data[setting]
        # Loading data
        epochs = setting_data["epochs"]
        N_c = setting_data["num_examples"]
        s = setting_data["sampling_batch"]
        sigma = setting_data["noise_multiplier"]
        delta = 1/N_c
        learning_rate = setting_data["learning_rate"]
        print ("Loading setting: %s" % setting)
        json_output = setting_data
        data_path = "./graphs/data/" + setting_file_name + '/appendix'
        isExist = os.path.exists(data_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(data_path)
            print("The new directory is created: %s" % data_path)
        K = N_c * epochs
        T = K/s
        print(compute_eps_from_sigma_and_delta(sigma,delta,K,N_c,s))
        print(compute_eps_uniform(epochs, sigma, N_c, s, delta))
# if __name__ == "__main__":
#     # N_c,K,delta,s = setting_3()
#     setting_file_name = "settings_sample_size_exp"
#     settings = ["setting_1","setting_2","setting_3","setting_4","setting_5","setting_6"]
#     with open("./%s.json" % setting_file_name, "r") as json_file:
#         json_data = json.load(json_file)
#
#     json_output_test = json_data
#     for setting in settings:
#         setting_data = json_data[setting]
#         # Loading data
#         epochs = setting_data["epochs"]
#         N_c = setting_data["num_examples"]
#         s = setting_data["sampling_batch"]
#         delta = setting_data["delta"]
#         print ("Loading setting: %s" % setting)
#         json_output = setting_data
#         data_path = "./graphs/data/" + setting_file_name + '/appendix'
#         isExist = os.path.exists(data_path)
#
#         if not isExist:
#             # Create a new directory because it does not exist
#             os.makedirs(data_path)
#             print("The new directory is created: %s" % data_path)
#         eps_dpsgd_arr = []
#         sigma_arr = []
#         eps_fdp_arr = []
#         mu_fdp_arr = []
#         print("Setting %s loaded" % setting)
#         for epoch in range(1,epochs+1):
#             # print("epoch: %d" % epoch)
#             K = N_c * epoch
#             # if(N_c != None):
#             eps_dpsgd, sigma = DP_calculator_CSS(N_c, K, delta, s)
#             T = K/s
#             if(eps_dpsgd != None):
#                 # print("-"*20)
#                 # print("N_c=",N_c)
#                 # print("K=",K)
#                 # print("epsilon=",eps)
#                 # print("delta=",delta)
#                 # print("sigma=",sigma)
#                 # print("T=",T)
#                 # print("s_i=",s)
#                 # print("-"*20)
#                 eps_dpsgd_arr.append(eps_dpsgd)
#                 sigma_arr.append(sigma)
#
#
#                 # Compute actual eps-fdp value
#                 # eps_fdp = compute_eps_poisson(epoch, sigma, N_c, s, delta)
#                 # mu_fdp = compute_mu_poisson(epoch, sigma, N_c, s)
#                 eps_fdp = compute_eps_uniform(epoch, sigma, N_c, s, delta)
#                 mu_fdp = compute_mu_uniform(epoch, sigma, N_c, s)
#                 eps_fdp_arr.append(eps_fdp)
#                 mu_fdp_arr.append(mu_fdp)
#             else:
#                 print("invalid eps epoch:",epoch)
#         json_output["sigma"]= sigma_arr
#         json_output_test["sigma"]= sigma_arr
#         # input(json_output_test["setting_1"]["sigma"])
#         json_output["eps_dpsgd"] = eps_dpsgd_arr
#
#         json_output["eps_fdp"] = eps_fdp_arr
#         json_output["mu_fdp"]= mu_fdp_arr
#         with open(data_path + '/' + setting + '.json', "w") as data_file:
#             json.dump(json_output, data_file)
#             # N_c = 60000
#     with open('./settings_main_theorem(test).json', "w") as data_file:
#         # input(json_output_test["setting_6"])
#         json.dump(json_output_test, data_file)
#     # delta = 1/N_c
#     # eps_arr = [i/1000 for i in range(100,1000)] # 0.01 -> 1.0
#     # gamma = []
#     # gamma_alphasquare = []
#     # gamma_alphasquare_mins = []
#     # gamma_alphasquare_mins_eps = []
#     # test = []
#     # sigmas = []
#     # GAM_item = 0
#     # tmp = 0
#     # for eps in eps_arr:
#     #     sigma = np.sqrt(2*np.log(1/delta)/eps)
#     #     # print(item)
#     #     # print(sigma)
#     #     alpha_bar = [i/1000 for i in range(1000)] # 0.01 -> 0.5
#     #     # print(alpha_bar)
#     #     GAM = np.inf
#     #     x = 0
#     #     for item in alpha_bar:
#     #         g = Compute_gamma(item,sigma)
#     # if(sigma <= 2*np.exp(1)*np.sqrt(item)/(1-item)):
#     #             continue
#     #         # print(g)
#     #         # print(g/pow(item,2))
#     #         # gamma.append(g)
#     #         if (g < 0):
#     #             continue
#     #             # input()
#     #         if(GAM > g/pow(item,2)):
#     #             GAM = g/pow(item,2)
#     #             GAM_item = item
#     #             x = item
#     #             # input(GAM)
#     #     # if(eps >= 0.5):
#     #     #     print("GAM",GAM)
#     #     #     print("alpha_bar",x)
#     #     #     print("gamma", Compute_gamma(x,sigma))
#     #     #     print("sigma",sigma)
#     #     #     input()
#     #     # if(tmp < GAM*eps):
#     #     #     tmp = GAM*eps
#     #     # else:
#     #     #     print("eps_drop",eps)
#     #     #     print("alpha_bar_drop",x)
#     #     #     input()
#     #     test.append(GAM_item)
#     #     sigmas.append(sigma)
#     #     gamma_alphasquare_mins.append(GAM)
#     #     gamma_alphasquare_mins_eps.append(GAM*eps)
#     # fig, axes = plt.subplots(nrows=2, ncols=2)
#
#     # axes[0][0].plot(eps_arr, gamma_alphasquare_mins)
#     # axes[0][0].set_title('N_c = 60000, delta = 1/N_c')
#     # axes[0][0].set_xlabel('eps')
#     # axes[0][0].set_ylabel('gamma/alpha_bar^2')
#
#     # axes[0][1].plot(eps_arr, gamma_alphasquare_mins_eps)
#     # axes[0][1].set_title('N_c = 60000, delta = 1/N_c')
#     # axes[0][1].set_xlabel('eps')
#     # axes[0][1].set_ylabel('eps*gamma/alpha_bar^2')
#
#     # axes[1][1].plot(sigmas, test)
#     # axes[1][1].set_title('N_c = 60000, delta = 1/N_c')
#     # axes[1][1].set_xlabel('sigma')
#     # axes[1][1].set_ylabel('alpha_bar')
#     # eps = 0.564
#     # sigma = np.sqrt(2*np.log(1/delta)/eps)
#     # alpha_bar = [i/1000 for i in range(450)]
#     # a = []
#     # for item in alpha_bar:
#     #     g = Compute_gamma(item,sigma)
#     #     if(sigma <= 2*np.exp(1)*np.sqrt(item))/(1-item):
#     #         continue
#     #     # print(g)
#     #     # print(g/pow(item,2))
#     #     # gamma.append(g)
#     #     if (g < 0):
#     #         continue
#     #     a.append(item)
#     #     gamma_alphasquare.append(g/pow(item,2))
#     # axes[1][0].plot(a, gamma_alphasquare)
#     # axes[1][0].set_title('eps = 0.8, N_c = 60000, delta = 1/N_c')
#     # axes[1][0].set_xlabel('alpha_bar')
#     # axes[1][0].set_ylabel('gamma/alpha_bar^2')
#     # plt.show()
