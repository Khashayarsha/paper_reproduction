#%%
import pandas as pd
import numpy as np
from pydantic import constr
import bivariate_poisson
import os
from scipy import optimize
from scipy.optimize import minimize
from scipy import stats
import scipy
from TeamTracker import TeamTracker
import pickle as pkl
import time

os.chdir(r"c:/Users/XHK/Desktop/thesis_code/paper_reproduction/src/models/")
print(f'current directory = {os. getcwd()}')
#os.chdir('./src/models')
data_path = r"../../data/interim"


def get_initialized_data(data_location=data_path, data_type=".pkl", data_name="df_initialized"):
    if data_type == ".pkl":
        data_file = data_name + data_type
        print('attempting to retrieve data from: ', data_location)
        os.chdir(data_location)
        data = pd.read_pickle(data_file)
    print("succesfully retrieved data from: ",
          data_location, "... , data = ", data_file)
    return data


df = get_initialized_data()

barmad = df[(df.HomeTeamCat.isin({5})) & (df.AwayTeamCat.isin({21})) | (
    df.AwayTeamCat.isin({5})) & (df.HomeTeamCat.isin({21}))]

maher_estimates = [0.3515203868747915, 0.1650621878904502, 0.0241674176480724,
                           -0.005631502146021409]

alpha_barca, alpha_madrid, beta_barca, beta_madrid = maher_estimates
f1 = maher_estimates.copy()

team_amount = 2
def construct_w(f1, b1, b2):
    f1_values =f1.reshape((team_amount*2, 1))
    ones_vec = np.ones((team_amount*2, 1))
    diagonal_of_B = np.array(
        ([b1]*team_amount, [b2]*team_amount)).reshape((team_amount*2, 1))

    rhs = ones_vec - diagonal_of_B
    result = np.multiply(f1_values, rhs)
    return result.reshape(team_amount*2,)


def update_step(fijt,x,y w, psi):
    A1,A2, B1, B2,  l3, delta = psi
    ai, aj, bi, bj = fijt
    s_ai, s_aj, s_bi, s_bj = bivariate_poisson.score(fijt, x, y, l3, delta) #scores
    w_ai, w_aj, w_bi, w_bj = w
    ai_next=w_ai + B1 * ai + A1 * s_ai
    aj_next=w_aj + B1 * aj + A1 * s_aj
    bi_next=w_bi + B2* bi + A2 * s_bi
    bj_next=w_bj + B2* bj + A2 * s_bj

    return np.array([ai_next, aj_next,bi_next, bj_next])
