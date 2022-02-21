#%%
import bivariate_poisson
import os 
import pandas as pd 
import numpy as np 
import time 
os.chdir(r"c:/Users/XHK/Desktop/thesis_code/paper_reproduction/src/models/")
print(f'current directory = {os. getcwd()}')
#os.chdir('./src/models')
data_path = r"../../data/interim"
print("Expected directory containing data = ", data_path)
os.chdir(data_path)
tic = time.perf_counter()

# Takes the df_initialized.pkl file with the Maher-initialized vector and runs the score-driven model on it
# Output is a matrix of f_t vectors, which are all the estimated team-strenghts at time t
# Output saved in data/processed
# this matrix can then be used to predict win-probabilities one step ahead.


#get df_initialized.pkl

def get_trained_data(data_location=data_path, data_type=".pkl", data_name="df_trained"):
    if data_type == ".pkl":
        data_file = data_name + data_type
        print('attempting to retrieve data from: ', data_location)
        os.chdir(data_location)
        data = pd.read_pickle(data_file)
    print("succesfully retrieved data from: ",
          data_location, "... , data = ", data_file)
    return data

df = get_trained_data()


    

def forecast(ft_total, matches, num_teams,df):
    df['forecast'] = 0 
    # input: ft_total = matrix (num_teams  x  num_rounds)   of strenghts based on all games until time t 
    #        matches = vector of tuples (i,j) ~ (home_team,away_team)
    # output: vector of predictions for all matches, "N/A" elsewhere

    #create l1ij, l2ij for all teams in matches
    return 0


def prob_home_win(row):
    ai, aj, bi, bj, l3, delta = row["ai aj bi bj lambda delta".split()]
    l1, l2 = bivariate_poisson.link_function(ai, aj, bi, bj, delta)
    probability = 0
    for x in range(1, 25):
        for y in range(0,x):
            probability+= bivariate_poisson.pmf(x,y,l1,l2,l3)
    return probability

def prob_draw(row):
    ai, aj, bi, bj, l3, delta = row["ai aj bi bj lambda delta".split()]
    l1, l2 = bivariate_poisson.link_function(ai, aj, bi, bj, delta)
    probability = 0
    for x in range(1, 25):
            y=x
            probability += bivariate_poisson.pmf(x, y, l1, l2, l3)

    return probability
def prob_home_loss(row):
    ai, aj, bi, bj, l3, delta = row["ai aj bi bj lambda delta".split()]
    l1, l2 = bivariate_poisson.link_function(ai, aj, bi, bj, delta)
    probability = 0
    for y in range(1, 25):
        for x in range(0, y):
            probability += bivariate_poisson.pmf(x, y, l1, l2, l3)
    return probability
    
