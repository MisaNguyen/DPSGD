from accountants.gdp_accountant_tensor import compute_mu_uniform, eps_from_mu
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def compute_f1(eps,delta,alpha):
    return 1-delta-np.exp(eps)*alpha
def compute_f2(eps,delta,alpha):
    return np.exp(-eps)*(1-delta-alpha)
def compute_f(f1,f2):
    return [max(0, f1[i], f2[i]) for i in range(len(f1))]

def compute_G(alpha,mu):
    return stats.norm.cdf(stats.norm.ppf(1-alpha) - mu)

def compute_h_sigma(sigma):
    term_2 = np.exp(pow(-sigma,-2))*stats.norm.cdf(3*pow(sigma,-1)/2)
    term_3 = 3* stats.norm.cdf(pow(-sigma,-1)/2)
    return np.sqrt(term_2 + term_3 - 2)
def main():
    use_fdp = False
    epoch = 50
    noise_multi = 0.01875
    N = 45000 # Cifar 10
    batch_size = 64
    delta = 1/N
    m = N/batch_size
    # sigma = 0.01875

    # print("Print1",3* stats.norm.cdf(pow(-sigma,-1)/2))
    # test = compute_h_sigma(sigma)
    # input(test)
    # gdp_mu = compute_mu_uniform(epoch, noise_multi, N, batch_size)
    gdp_mu = np.sqrt(m/N)/noise_multi
    our_mu = 0.38
    print("their",gdp_mu)
    print("our", our_mu)
    #------------------------------#
    alpha = np.linspace(0,1,100)
    plt.title("N = %s, m = %s, sigma = %s, epoch = %s" % (N,batch_size,noise_multi,epoch))
    #------------------------------#
    indinguishable_line = 1-alpha
    plt.plot(alpha, indinguishable_line, '-g', label='indinguishable_line')
    if (use_fdp):
        gdp_eps = eps_from_mu(gdp_mu, delta)
        our_eps = eps_from_mu(our_mu, delta)
        #------------------------------#
        f1 = compute_f1(gdp_eps,delta,alpha)
        f2 = compute_f2(gdp_eps,delta,alpha)
        f = compute_f(f1,f2)
        # print("f",f)
        # plt.plot(alpha, f, '-r', label='dong el at')
        #------------------------------#
        f1 = compute_f1(our_eps,delta,alpha)
        f2 = compute_f2(our_eps,delta,alpha)
        f = compute_f(f1,f2)
        # print("f",f)
        plt.plot(alpha, f, '-b', label='our')
        #------------------------------#
    else:
        #------------------------------#
        # G = compute_G(alpha,gdp_mu)
        # print("f",f)
        # plt.plot(alpha, G, '-r', label='G_((m/N)/sigma)')
        #------------------------------#
        G = compute_G(alpha,our_mu)
        # print("f",f)
        plt.plot(alpha, G, '-b', label='G_%s' %(our_mu))
        #------------------------------#

    plt.legend()
    plt.show()
if __name__ == '__main__':
    main()