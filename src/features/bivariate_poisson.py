import numpy as np
import pandas as pd
import os
import scipy.special
factorial = scipy.special.factorial
ncr = scipy.special.comb


def pmf(x, y, l1, l2, l3, n=1):  #bivariate poisson
    # returns probability mass for a given score (x,y), given lambdas l1, l2, l3
    # check if x and y are integers:
    #print(x, y)
    minimum = int(min(x, y))    
    # print(minimum)
    first = (np.exp(-1*(l1+l2+l3)) * ((l1**x)/factorial(x))
             * ((l2**y)/factorial(y)))

    def sum_part(mini, x, y):
        # print(mini)
        # print(type(mini))
        total = 0
        if mini == 0:
            k = 0
            total += ncr(x, k, exact=True) * ncr(y, k, exact=True) * \
                factorial(k) * ((l3/(l1*l2))**k)
        else:
            for k in range(0, int(mini)):
                total += ncr(x, k, exact=True) * ncr(y, k, exact=True) * \
                    factorial(k) * ((l3/(l1*l2))**k)
        return total
    second = sum_part(int(minimum), x, y)

    return first*second


def S(q, ft, psi):
    # helperfunction for gradient
    # part of bivar. poisson gradient, see appendix at p.32 of Lit2017

    l1, l2, l3 = psi
    min_xy = min(x, y)
    sum = 0
    for k in range(0, min_xy):
        sum += ncr(x, k, exact=True) * ncr(y, k, exact=True) * \
            factorial(k) * (k**q) * ((l3/(l1*l2))**k)
    return sum


def U(ft, psi, x, y):
    # helper-function for gradient
    # part of bivar. poisson gradient, see appendix at p.32 of Lit2017

    return S(1, ft, psi, x, y) / S(0, ft, psi, x, y)


def score(x, y, l1, l2, l3):  # function used to be called 'gradient'
    # part of bivar. poisson gradient, see appendix at p.32 of Lit2017
    # returns the dlog(p)/d_fijt
    return np.array((x-l1 - U(ft, psi),
                     y-l2 - U(ft, psi),
                     l2-y + U(ft, psi),
                     l1-x + U(ft, psi))).T
