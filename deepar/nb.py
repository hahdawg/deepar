import numpy as np


def nbr(alpha, mu):
    p = alpha*mu/(1 + alpha*mu)
    r = 1/alpha
    print(p, r, p*r/(1 - p))
    res = np.random.negative_binomial(n=r, p=p, size=1000000)
    return res


def rescale(alpha, mu, nu):
    mu *= nu
    alpha /= nu
    return nbr(alpha, mu)
