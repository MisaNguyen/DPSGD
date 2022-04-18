import numpy as np
import json
import os
from gdp_accountant_tensor import compute_eps_poisson,compute_mu_poisson, compute_eps_uniform, compute_mu_uniform
# Note: This is the constant sample size sequence size only !!!
def compute_gamma(alpha_bar,sigma):
    C_1 = pow(2,4)*alpha_bar/(1-alpha_bar) # 2^4*alpha/(1-alpha)
    C_2 = sigma/pow(1-np.sqrt(alpha_bar),2) # sigma/(1-sqrt(alpha))^2
    C_3 = (1/(sigma*(1-alpha_bar) - 2 * np.exp(1)* np.sqrt(alpha_bar)))*np.exp(3)/sigma # 1/(sigma*(1-alpha))
    C_4 = np.exp(3/pow(sigma,2))
    C_5 = 2/(1-alpha_bar)
    # print("C_1",C_1)
    # print("C_2",C_2)
    # print("C_3",C_3)
    # print("C_4",C_4)
    # print("C_5",C_5)
    return C_1*(C_2+C_3)*C_4 + C_5

def compute_alpha_bar(eps,N_c,K,gamma):
    return eps*N_c/(gamma*K) # N_c/K = k

def check_k_condition(eps,N_c,K,gamma,delta,theta = 1): # Theta = 1 because s_max = s_bar (sample_size sequence ???)
    return K / N_c >= np.sqrt(2 * eps * np.log(1 / delta)) * np.exp(1) / (gamma * theta)

def check_T_condition(eps, N_c,K,gamma,T,theta=1):
    return T >= (gamma * theta * theta) / eps * pow(K / N_c, 2)

def compute_T(eps, N_c,K,gamma,theta=1):
    return (gamma * theta * theta) / eps * pow(K / N_c, 2)

def compute_T_conjecture(eps, N_c,K):
    return pow(K/N_c,2)/(4*eps)

def compute_eps_from_sigma(sigma,delta):
    # sigma^2*eps = 2(eps+ln(1/delta))
    # eps (sigma^2 - 2 ) = 2ln(1/delta)
    # eps = 2ln(1/delta)/(sigma^2-2)
    if (sigma*sigma < 2):
        print("Sigma must be greater than: ", np.sqrt(2))
        return None
    return 2*np.log(1/delta)/ (sigma*sigma -2)  # Assuming sigma^2 > 2 *** important

def compute_eps_from_gamma(T,gamma,theta,N_c,K):
    k = K/N_c
    return (gamma*pow(theta,2)*pow(k,2))/T

# Binary search for sigma
def compute_optimal_eps_sigma(sigma_left,sigma_right,theta,epoch,T,delta,N_c,K):
    alpha_bar = theta * theta * epoch/T
    sigma_mid = (sigma_left+sigma_right)/2
    print("sigma_mid", sigma_mid)
    eps_left_1 = compute_eps_from_sigma(sigma_left,delta)
    gamma_left = compute_gamma(alpha_bar,sigma_left)
    eps_left_2 = compute_eps_from_gamma(T,gamma_left,theta,N_c,K)
    print("eps_left_1",eps_left_1)
    print("eps_left_2",eps_left_2)
    print("k_condition_eps_left_1", check_k_condition(eps_left_1,N_c,K,gamma_left,delta))
    print("k_condition_eps_left_2", check_k_condition(eps_left_2,N_c,K,gamma_left,delta))
    eps_mid_1 = compute_eps_from_sigma(sigma_mid,delta)
    gamma_mid = compute_gamma(alpha_bar,sigma_mid)
    eps_mid_2 = compute_eps_from_gamma(T,gamma_mid,theta,N_c,K)
    print("eps_mid_1",eps_mid_1)
    print("eps_mid_2",eps_mid_2)
    print("k_condition_eps_mid_1", check_k_condition(eps_mid_1,N_c,K,gamma_mid,delta))
    print("k_condition_eps_mid_2", check_k_condition(eps_mid_2,N_c,K,gamma_mid,delta))
    eps_right_1 = compute_eps_from_sigma(sigma_right,delta)
    gamma_right = compute_gamma(alpha_bar,sigma_right)
    eps_right_2 = compute_eps_from_gamma(T,gamma_right,theta,N_c,K)
    print("eps_right_1",eps_right_1)
    print("eps_right_2",eps_right_2)
    print("k_condition_eps_right_1", check_k_condition(eps_right_1,N_c,K,gamma_right,delta))
    print("k_condition_eps_right_2", check_k_condition(eps_right_2,N_c,K,gamma_right,delta))
# def compute(sigma,delta, N_c,K,T):
#     eps = compute_eps_from_sigma(sigma, delta)
#     if (eps == None):
#         return None
#     gamma = 0
#     gamma_new = 2 # gamma start with 2
#     print("Finding smallest gamma value")
#     while gamma_new - gamma > 1e-5*gamma:
#         gamma = gamma_new
#         alpha_bar = compute_alpha_bar(eps,N_c,K,gamma)
#         gamma_new = compute_gamma(alpha_bar,sigma)
#         # Check whether the new gamma still satisfies the condition (2) and (3)
#         if(not check_k_condition(eps,N_c,K,gamma_new,delta) or not check_T_condition(eps, N_c, K, gamma_new, T)):
#             break
def Compute_alpha_bar(eps,N_c,K,gamma):
    return eps*N_c/(gamma*K) # N_c/K = 1/k

if __name__ == "__main__":
    # N_c,K,delta,s = setting_3()
    # setting_file_name = "settings_main_theorem(test)"
    # setting_file_name = "settings_sample_size_exp"
    setting_file_name = "settings_ICML_table_2"
    # setting_file_name = "settings_ICML_N_1000000"
    # settings = ["setting_1","setting_2","setting_3","setting_4","setting_5","setting_6","setting_7"]
    settings = ["setting_1","setting_2","setting_3","setting_4"]
    with open("./%s.json" % setting_file_name, "r") as json_file:
        json_data = json.load(json_file)
    for setting in settings:
        print("-"*20)
        print ("Loading setting: %s" % setting)
        setting_data = json_data[setting]
        # Loading data
        epochs = setting_data["epochs"]
        N_c = setting_data["num_examples"]
        # s = setting_data["sampling_batch"]
        sigma = setting_data["noise_multiplier"]
        delta = 1/N_c
        learning_rate = setting_data["learning_rate"]
        # print(len(sigma_arr))
        # print("epoch,",epochs)
        # input()
        theta = 1


        json_output = setting_data
        data_path = "./graphs/data/" + setting_file_name + '/mainbody'
        isExist = os.path.exists(data_path)

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(data_path)
            print("The new directory is created: %s" % data_path)
        eps_dpsgd_arr = []
        # sigma_arr = []
        eps_fdp_arr = []
        mu_fdp_arr = []
        print("Setting %s loaded" % setting)

        # print("epoch: %d" % epoch)
        K = N_c * epochs # Total iterations = dataset size * epochs
        # T = K/s # number of rounds = total iterations / constant_   sample_size

        # print(epoch)

        ### Main calculations ###
        # alpha_bar = theta * theta * epochs/T # Note: take T = theta^2/alpha *k
        # gamma = compute_gamma(alpha_bar,sigma)
        gamma = 0 #initial gamma
        new_gamma = 2 #initial gamma
        # print("gamma:",gamma)
        eps_dpsgd = compute_eps_from_sigma(sigma,delta)
        print('eps_dpsgd:', round(eps_dpsgd,4))
        # fdp_eps = compute_eps_uniform(epochs, sigma, N_c, s, delta)
        while(new_gamma - gamma >= 0.0001 * gamma):
            gamma = new_gamma
            alpha_bar = Compute_alpha_bar(eps_dpsgd,N_c,K,gamma)
            new_gamma = compute_gamma(alpha_bar,sigma)
        print("alpha_bar=", alpha_bar)
        print("gamma =", gamma)
        if(not check_k_condition(eps_dpsgd,N_c,K,gamma,delta)):
            print("k must be larger than : ", np.sqrt(2 * eps_dpsgd * np.log(1 / delta)) * np.exp(1) / (gamma * theta))
            break
        T = compute_T(eps_dpsgd, N_c,K,gamma,theta=1)
        s_min_T = K/T
        print("s=", round(s_min_T,4))
        fdp_eps_1 = compute_eps_uniform(epochs, sigma, N_c, s_min_T, delta)
        print("fdp_eps", round(fdp_eps_1,4))
        print("factor", round(eps_dpsgd/ fdp_eps_1,4))
        T = compute_T_conjecture(eps_dpsgd, N_c,K)
        s_min_T_factor_4 = K/T
        print("s_conjecture=", round(s_min_T_factor_4,4))
        fdp_eps_2 = compute_eps_uniform(epochs, sigma, N_c, s_min_T_factor_4, delta)
        print("fdp_eps", round(fdp_eps_2,4))
        print("factor", round(eps_dpsgd/ fdp_eps_2,4))
        # compute_optimal_eps_sigma(sigma_left,sigma_right,theta,epoch,T,delta,N_c,K)
        # input()
        # eps_dpsgd = compute_eps_from_sigma(sigma,delta)

        # if(eps_dpsgd != None and check_k_condition(eps_dpsgd,N_c,K,gamma,delta)):
        #     print("our epsilon(main_body):", eps_dpsgd)
        #     print("fdp epsilon:", fdp_eps)
        #     if(eps_dpsgd != None):
        #         print("multi factor:", eps_dpsgd/fdp_eps)
        #     # print("-"*20)
        #     # print("N_c=",N_c)
        #     # print("K=",K)
        #     # print("epsilon=",eps)
        #     # print("delta=",delta)
        #     # print("sigma=",sigma)
        #     # print("T=",T)
        #     # print("s_i=",s)
        #     # print("-"*20)
        #     eps_dpsgd_arr.append(eps_dpsgd)
        #     # sigma_arr.append(sigma)
        #
        #
        #     # Compute actual eps-fdp value
        #     # eps_fdp = compute_eps_poisson(epoch, sigma, N_c, s, delta)
        #     # mu_fdp = compute_mu_poisson(epoch, sigma, N_c, s)
        #     eps_fdp = compute_eps_uniform(epochs, sigma, N_c, s, delta)
        #     mu_fdp = compute_mu_uniform(epochs, sigma, N_c, s)
        #     eps_fdp_arr.append(eps_fdp)
        #     mu_fdp_arr.append(mu_fdp)
        json_output["eps_dpsgd"] = eps_dpsgd_arr
        json_output["sigma"]= sigma
        json_output["eps_fdp"] = eps_fdp_arr
        json_output["mu_fdp"]= mu_fdp_arr
        with open(data_path + '/' + setting + '.json', "w") as data_file:
            json.dump(json_output, data_file)
    print("  & s\\_min\\_T = %f  & s\\_min\\_T\\_factor\\_4 = %f \\\\ \\hline" % (s_min_T,s_min_T_factor_4))
    print("Our DPA (main\_body) & %f & %f \\\\ \\hline" %(round(eps_dpsgd,4),round(eps_dpsgd,4)))
    print("Guassian DP accountant & %f & %f \\\\ \\hline" %(round(fdp_eps_1,4),round(fdp_eps_2,4)))
    print("Multiplication factor & %f & %f \\\\ \\hline" %(round(eps_dpsgd/ fdp_eps_1,4),round(eps_dpsgd/ fdp_eps_2,4)))
# """ OLD  MAIN()"""
# if __name__ == "__main__":
#     # N_c,K,delta,s = setting_3()
#     setting_file_name = "settings_main_theorem(test)"
#     settings = ["setting_1","setting_2","setting_3","setting_4","setting_5","setting_6"]
#     with open("./%s.json" % setting_file_name, "r") as json_file:
#         json_data = json.load(json_file)
#     for setting in settings:
#         print ("Loading setting: %s" % setting)
#         setting_data = json_data[setting]
#         # Loading data
#         epochs = setting_data["epochs"]
#         N_c = setting_data["num_examples"]
#         s = setting_data["sampling_batch"]
#         delta = setting_data["delta"]
#         sigma_arr = setting_data["sigma"]
#         # print(len(sigma_arr))
#         # print("epoch,",epochs)
#         # input()
#         theta = 1
#
#
#         json_output = setting_data
#         data_path = "./graphs/data/" + setting_file_name + '/mainbody'
#         isExist = os.path.exists(data_path)
#
#         if not isExist:
#             # Create a new directory because it does not exist
#             os.makedirs(data_path)
#             print("The new directory is created: %s" % data_path)
#         eps_dpsgd_arr = []
#         # sigma_arr = []
#         eps_fdp_arr = []
#         mu_fdp_arr = []
#         print("Setting %s loaded" % setting)
#         for epoch in range(1,epochs+1):
#             # print("epoch: %d" % epoch)
#             K = N_c * epoch
#             T = K/s
#             # print(epoch)
#             sigma = sigma_arr[epoch-1]
#             alpha_bar = theta * theta * epoch/T # Note: take T = theta^2/alpha *k
#             gamma = compute_gamma(alpha_bar,sigma)
#             eps_dpsgd = compute_eps_from_sigma(sigma,delta)
#
#             # compute_optimal_eps_sigma(sigma_left,sigma_right,theta,epoch,T,delta,N_c,K)
#             # input()
#             # eps_dpsgd = compute_eps_from_sigma(sigma,delta)
#
#             if(eps_dpsgd != None and check_k_condition(eps_dpsgd,N_c,K,gamma,delta)):
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
#                 # sigma_arr.append(sigma)
#
#
#                 # Compute actual eps-fdp value
#                 # eps_fdp = compute_eps_poisson(epoch, sigma, N_c, s, delta)
#                 # mu_fdp = compute_mu_poisson(epoch, sigma, N_c, s)
#                 eps_fdp = compute_eps_uniform(epoch, sigma, N_c, s, delta)
#                 mu_fdp = compute_mu_uniform(epoch, sigma, N_c, s)
#                 eps_fdp_arr.append(eps_fdp)
#                 mu_fdp_arr.append(mu_fdp)
#         json_output["eps_dpsgd"] = eps_dpsgd_arr
#         json_output["sigma"]= sigma_arr
#         json_output["eps_fdp"] = eps_fdp_arr
#         json_output["mu_fdp"]= mu_fdp_arr
#         with open(data_path + '/' + setting + '.json', "w") as data_file:
#             json.dump(json_output, data_file)
