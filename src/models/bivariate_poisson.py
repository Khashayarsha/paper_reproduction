import numpy as np
import pandas as pd
import os
import scipy.special
factorial = scipy.special.factorial
ncr = scipy.special.comb
 
def link_function(ai, aj, bi, bj, delta, thresh_min = -20, thresh_max = 20): #threshs waren eerst -3.22, 3.22
    # 0.04 and 3.22 are log(e^(1/25)) and log(25)
    exponent = np.clip([(ai - bj + delta), (aj-bi)], a_min = thresh_min, a_max = thresh_max)
    l1, l2 = np.e**exponent
    #np.clip([np.exp(ai - bj + delta), np.exp(aj - bi)], a_min=thresh_min, a_max=thresh_max)
    return l1, l2


def pmf(x, y, l1, l2, l3, n=1, log = False):  #bivariate poisson
    # returns probability mass for a given score (x,y), given lambdas l1, l2, l3
    # check if x and y are integers:
    #print(x, y)
    x,y = int(x), int(y)
    minimum = int(min(x, y))  
    #print('minimum in bivariate pmf, and type: ', minimum, type(minimum))
    # print(minimum)
    first = (np.exp(-1*(l1+l2+l3)) * ((l1**x)/factorial(x))
             * ((l2**y)/factorial(y)))
    # print(f"x = {x}, y={y}, l1 = {l1}, l2 = {l2}, l3= {l3}")
    def sum_part(mini, x, y):
        # print(mini)
        # print(type(mini))
        total = 0
        # if mini == 0:
        #     k = 0
        #     total += ncr(x, k, exact=True) * ncr(y, k, exact=True) * \
        #         factorial(k) * ((l3/(l1*l2))**k)
        # else:
        for k in range(0, int(mini)+1):
            total += ncr(x, k, exact=True) * ncr(y, k, exact=True) * \
                factorial(k) * ((l1*l2)**(-k)) *  (l3**k)  # ((l3/(l1*l2))**k)
        return total
    second = sum_part(int(minimum), x, y)
    #first, second = np.clip([first, second], a_min = 0.01, a_max = 200)
    if log == True: 
        # + 2000 #= np.log(first*second)    np.log(first) + np.log(second)
        return np.log(first) + np.log(second ) #+ np.log(second )
    return first*second


def S(q, psi, x , y):
    # helperfunction for gradient
    # part of bivar. poisson gradient, see appendix at p.32 of Lit2017
    #print('x and y ', x, y, type(x), type(y))
    l1, l2, l3 = psi
    x, y = int(x), int(y)
    min_xy = min(x, y)
    #print('min_xy in S from bivariate_poisson. Value and type: ', min_xy, type(min_xy))
    sum = 0

    
    for k in range(0, min_xy+1): 
        sum += ncr(x, k, exact=True) * ncr(y, k, exact=True) * \
            factorial(k) * (k**q) * ((l1*l2)**(-k)) * \
            (l3**k)  # ((l3/(l1*l2))**k)

    
    
#print(f'S(q, psi, x , y) is using parameters: q={q} psi= {psi} x= {x}y = {y} and result = {sum}')

    return sum


def U(psi, x, y):
    # helper-function for gradient
    # part of bivar. poisson gradient, see appendix at p.32 of Lit2017
    #print('psi: ', psi)
    #print('x,y', x,y)
    #print('S(1, ..) , S(0,...) values: ', S(1, psi, x, y), S(0, psi, x, y))
    return S(1, psi, x, y) / S(0, psi, x, y)


def score(fijt,x,y,l3, delta):
    ai, aj, bi, bj = fijt 
    l1,l2 = link_function(ai, aj, bi, bj, delta) 
    #p.7 of Koopman/Lit 2019 says "in practice the infinite upper bound
                    #for l1ij and l2ij is replaced by 25"
    # if (l1 > 5 or l2 > 5):
    #     print("l1, l2 are VERRYYY HIGHHH: VALUES: ", l1, l2)

    psi = (l1,l2,l3)

    
    return np.array((x-l1 - U(psi,x,y),
                     y-l2 - U(psi,x,y),
                     l2-y + U(psi,x,y),
                     l1-x + U(psi,x,y))).T

    
# def score(x, y, l1, l2, l3):  # function used to be called 'gradient'
#     # part of bivar. poisson gradient, see appendix at p.32 of Lit2017
#     # returns the dlog(p)/d_fijt
#     l1, l2 = link_function(fijt)
#     return np.array((x-l1 - U(ft, psi),
#                      y-l2 - U(ft, psi),
#                      l2-y + U(ft, psi),
#                      l1-x + U(ft, psi))).T

def double_poisson_pmf(x, y, l1, l2, l3, n=1, log=False):  
    x,y = int(x), int(y)
    if not log:
        p =  (np.e**-l1) * (l1**x)*(np.e**-l2) * (l2**y) / (factorial(x) *factorial(y))
        return p
    if log:
        first = -l1 + x*np.log(l1) - np.log(factorial(x)) 
        second = -l2 + y*np.log(l2) - np.log(factorial(y))
        return first+second
    

def double_poisson_score(fijt, x, y,  delta):
    ai, aj, bi, bj = fijt 
    
    l1, l2 = link_function(ai, aj, bi, bj, delta )

    return np.array((x-l1  ,
              y-l2  ,
              l2-y  ,
              l1-x  )).T
