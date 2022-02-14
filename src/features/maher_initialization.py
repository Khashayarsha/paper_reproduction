# %%
from scipy import optimize
from scipy import stats
import scipy
import bivariate_poisson
#from score_driver import ScoreDriver
import numpy as np
import pandas as pd
import os
import scipy.special
factorial = scipy.special.factorial
ncr = scipy.special.comb

#in:   prepped_data.pkl from preprocess_final.py
#out:  f1.pkl       the power_vector of the first year, for use in
#      saves it in ../../data/interim  folder as df_initialized




if os.getcwd().split('\\')[-1] != 'features':
    os.chdir(r"C:/Users/XHK\Desktop/thesis_code/paper_reproduction/src/features")
script_folder = os.getcwd()
data_folder = r"../../data/interim"

print("Expected directory containing data = ", data_folder)
os.chdir(data_folder)

def get_first_year(df, test = True):
    if test==True:
        "using default first_year value of 380 for testing"
        first_year = df[:380]
        return first_year

name = "prepped_data_1_2_15_29_25.pkl"
df = pd.read_pickle(name)
first_year = get_first_year(df)

# def update_round(current_t):
#     if current_t == 0:
#         f_t, absentees = df['absentees'].iloc[0]
#         current_t += 1
#     else:
#         # update for all teams in round t = current_t:
#         teams = teams_in_round(current_t)
#         for (i, j) in teams:
#             print('lel')

#     return current_t


def num_unique_teams(data, which="False"):
    if which == False:
        return len(list(set(data[["HomeTeamCat", "AwayTeamCat"]].values.flatten())))
    else:
        number = len(
            list(set(data[["HomeTeamCat", "AwayTeamCat"]].values.flatten())))
        which_teams = set(
            data[["HomeTeamCat", "AwayTeamCat"]].values.flatten())
        return number, which_teams


def constraint_alphas(x):
    # hardcoded for convenience.
    absentees = [1, 2, 7, 10, 11, 13, 14, 17, 19, 22, 26, 27, 31]
    """ Sum of alphas (attack strengths) should be 0 for identification purposes. """
    alphas = x[:33]
    alphas[absentees] = 0  # strengths of non participants have to be 0
    return -1 * np.sum(alphas)


def constraint_delta(x):
    delta = x[-1]

    return delta


def constraint_gamma(x):
    gamma = x[-2]
    return gamma


def maher_initialization(first_year_data):
    num_first_year, which_first_year = num_unique_teams(
        first_year, which=True)

    length = (len(num_unique_teams(df)[1]) * 2 + 2)
    print("length =", length)
    theta_ini = np.zeros(length)
    print(theta_ini.shape)

    con1 = {'type': 'eq', 'fun': constraint_alphas}
    con2 = {'type': 'ineq', 'fun': constraint_delta}
    con3 = {'type': 'ineq', 'fun': constraint_gamma}
    cons = [con1,con2, con3]
    bound = (-1.0, 1.0)
    boundaries = [(-np.inf, np.inf) if i < length -
                  2 else (0, 1) for i in range(length)]
    options = {'eps': 1e-09,  # was 1e-09
               'disp': True,
               'maxiter': 500}
    results = scipy.optimize.minimize(log_likelihood_maher, theta_ini, args=(first_year),
                                      options=options,
                                      method='SLSQP',
                                      constraints=cons,
                                      bounds=boundaries)  # restrictions in parameter space
    print(results)
    return results


def log_likelihood_maher(theta, data, minimize=True):
    n = 33
    # print(x)
    alphas = theta[:n]
    betas = theta[n:2*n]
    delta = theta[-1]  # 0.3  # theta[2*n:(2*n)+1]  # home-ground advantage
    gamma = theta[-2]  # 0.05  # theta[(2*n)+1:]  # correlation
    total = 0

    for i in data.index:
        if i % 100 == 0:
            print("Analyzing game ", i)
        # print("GAME ", i)
        # print(type(x))
        # print(data)
        # print(data.shape)
        game = data.iloc[i]

        home = game["HomeTeamCat"]
        away = game["AwayTeamCat"]
        x, y = game["y"]
        l1 = np.exp(alphas[home] - betas[away] + delta)
        l2 = np.exp(alphas[away] - betas[home])
        l3 = gamma
        total += np.log(bivariate_poisson.pmf(x, y, l1, l2, l3))
    #total += 0.1 * np.sum(alphas)
    if minimize == True:
        print(f"log likelihood = {total} and when resturned to optimizer it is: {-1*total}")
        return -1*total
    return total


def initialize(data, model="1"):  # returns set of absentees (teams that didn't play in first year)
    if model == "1":
        size = 2 * num_unique_teams(df)
        num_first_year, set_first_year = num_unique_teams(
            first_year, which=True)  # teams that played first year
        _, set_all_teams = num_unique_teams(df, which=True)
        # teams that didn't play in first year, but did after
        #absentees = which_first_year.difference(all_teams)
        absentees = set_all_teams.difference(set_first_year)
    return absentees

mapping = df['mapping'].iloc[0]
result = maher_initialization(first_year)
strengths = [(mapping[i], result.x[i]) for i in range(33)]
strengths.sort(key=lambda x: x[1], reverse=True)
print("Strengths sorted: ")
print(strengths)
print("Gamma (correlation), delta (home team adv):",
      result.x[-2], result.x[-1])

# second initialization-method described in Lit2017
num_attacks, num_strengths = 33, 33


#f1 = result.x[:num_attacks+num_strengths]

f1 = pd.DataFrame(result.x, columns=["f1"])
teems = [mapping[i] for i in range(33)]*2
teems.append('gamma')
teems.append('delta')
f1["teams"] = teems
f1.to_csv("f1_maher_init_24january.csv")
f1.to_pickle("f1_maher_init.pkl")

f_one = np.zeros_like(df['participants'])
f_one[:len(f1['f1'])] = f1['f1']

df['f1'] = f_one

print('trying to pickle df_initialized, saving in data/processed')
os.chdir('../interim')
df.to_pickle('df_initialized2.pkl')
print('testing pickling')
pd.read_pickle("df_initialized2.pkl")


print('done producing maher_initialization, saved in ', os.getcwd())
# %%
