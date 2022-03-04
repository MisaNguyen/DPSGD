from PyTorchProjectFramework.accountants.gdp_accountant import GaussianAccountant


if __name__ == '__main__':

    accountant = GaussianAccountant()
    steps = 10
    delta = pow(10,-6)
    print(delta)
    for i in range(steps):
        accountant.step(noise_multiplier = 0.5, sample_rate=0.5)
        print(accountant.get_epsilon(delta=delta, poisson=False))

    pass