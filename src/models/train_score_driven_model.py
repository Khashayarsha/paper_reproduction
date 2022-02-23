# %% 
# %reset -f
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

from decorators import CallCountDecorator

np.seterr(over='raise')
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

def get_initialized_data(data_location = data_path, data_type = ".pkl", data_name = "df_initialized" ):
    if data_type == ".pkl":
        data_file =  data_name  + data_type
        print('attempting to retrieve data from: ', data_location)
        os.chdir(data_location)
        data = pd.read_pickle(data_file)
    print("succesfully retrieved data from: ", data_location, "... , data = ", data_file)
    return data 


df = get_initialized_data()

use_small_data = False
if use_small_data == True:
    print("USING SMALLER DATASET FOR TESTING")
    df = df.iloc[:700]



#import preprocess
team_amount = 33     #turn into a   function or do this in pre-process etc
f1_size = (team_amount * 2) + 2#+2    f_t just consists of the time-varying team atk and def str's. 
                            # DELTA AND GAMMA ARE NOT TIME VARYING IN THIS MODEL.
# psi1 = pd.read_csv("f1_maher_init2.csv")
# print("TEST")
# df = pd.read_csv("processed_data2.csv")

#getting right parameters from initial f_vector (f1)

f1_init = df['f1'][:f1_size].values
#lambda_3 = f1.values[-2]  # noemde dit eerst gamma_1, maar is bs
delta_1, lambda_3 = f1_init[-1], f1_init[-2]
f1_init = f1_init[:f1_size-2] #ignores the delta and lambda_3


mapping = df['mapping'].iloc[0] #dictinary that maps    team_numbers -> teamnames.
inv_mapping = {v: k for k, v in mapping.items()}  #maps team_names -> team_numbers
unseen_teams_init = set(df['absentees'].iloc[0])
seen_teams_init = set(df.iloc[0].participants) 

#unseen_teams = set(df['absentees'].iloc[0])  #set of teams not playing in first round
#seen_teams = set(df.iloc[0].participants) #set of teams that played in first round (from which Maher init calculated)
team_tracker = TeamTracker(unseen_teams_init, seen_teams_init)

temp = [df[df["round_labels"] == t][["HomeTeamCat", "AwayTeamCat"]].values for t in set(df["round_labels"])]
# print("eval start")  #geen idee wat dit doet :') 05-01-2021
# df['participants'] = df.participants.apply(eval)
# print("eval end")
matches_per_round = [[(i, j) for i, j in round] for round in temp]
#turn into a dict for performance and correct indexing:
matches_per_round = {i+1: matches_per_round[i] for i in range(len(matches_per_round))} 


#seen_teams = set()
#unseen_teams = set(df["HomeTeamCat"].values).union(df["AwayTeamCat"].values)

def constrain_alphas(alpha_vector):
    #constrains alphas to summing to 1 ??
    return alpha_vector/(alpha_vector.sum(axis=0))

def construct_goals_dict():
    
    

    get_goals_dict = {}

    first_round = min(df['round_labels'].values)
    last_round  = max(df['round_labels'].values)
    round_numbers = range(first_round, last_round+1)

    for round in round_numbers:
        for match in matches_per_round.get(round):
            i, j = match
            match_results = df[(df["round_labels"] == round) & (
                df["HomeTeamCat"] == i) & (df["AwayTeamCat"] == j)]
            home_score = match_results.FTHG.values[0]
            away_score = match_results.FTAG.values[0]
            get_goals_dict.update({(i,j,round) : (home_score, away_score) })
    return get_goals_dict

def construct_participants_dict():

    part_dict = {}
    for round in set(matches_per_round.keys()):
        match_list  = matches_per_round.get(round)
        home_teams = [match[0] for match in match_list]
        away_teams = [match[1] for match  in match_list]
        participants = home_teams + away_teams

        part_dict.update({round: set(participants)})
    return part_dict


    # get_goals_dict = {(home, away, round): df[(df["round_labels"] == round) & (
    #     df["HomeTeamCat"] == home) & (df["AwayTeamCat"] == away)][['FTHG', 'FTAG']].values.flatten() for (home,away) in matches_per_round.get(round) for round in range(first_round, last_round+1)}

goal_dict = construct_goals_dict()
participants_dict = construct_participants_dict()

# def get_goals(i, j, t):
#     #
#     """ Returns home_goals, away_goals given two teams (integer category).
#     Returns a string if there exists no match between home and away in round t """

#     home, away, round = i, j, t

#     # verify team i plays team j in round t
#     matches_played = matches_per_round.get(round)
#     match_exists = (home,away) in matches_played
#     if not match_exists:
#         print(f"Match {(home,away)} in round {round} does not exist ")
#         return "get_goals method failed because MATCH DOESN'T EXIST"
#     else:
#         match = df[(df["round_labels"] == round) & (df["HomeTeamCat"] == home) & (df["AwayTeamCat"] == away)]
#         home_score = match.FTHG.values[0]
#         away_score = match.FTAG.values[0]
#     return home_score, away_score

# def get_home_goals(i,j,t):
#     home_goals, away_goals = goal_dict[i,j,t]
#     return home_goals

# def get_away_goals(i, j, t):
#     home_goals, away_goals = goal_dict[i,j,t]
#     return away_goals

 


def set_new_teams_parameters(new_teams, ft): 
    #initializes strenghts and defences to 0 for newly-appeared teams
    f = ft.copy()
    new_teams = list(new_teams)
    alpha_locs = new_teams
    beta_locs  = np.array(new_teams) + team_amount

    #print('ft in set_new_teams_parameters: ', ft)
    #print('alpha locs: ', alpha_locs)
    #print('beta_locs: ', beta_locs)

    f[alpha_locs] = 0
    f[beta_locs] = 0
    return f
def get_f1():
    return f1_init.copy()

def update_fijt(fijt,i,j,psi,wijt,t):
    a1, a2, b1, b2, l3, delta = psi
    alpha_i, alpha_j, beta_i, beta_j = fijt.copy()
    score = 0
    fijt_updated = fijt.copy() 
    
    # x = get_home_goals(i, j, t)
    # y = get_away_goals(i, j, t)
    x, y = goal_dict[i, j, t]
    
    
    score = bivariate_poisson.score(fijt.copy(),x, y, l3, delta) #must return a 4x1 vector
    

    Bij = np.diag([b1,b1,b2,b2])
    Aij = np.diag([a1,a1,a2,a2])

    fijt_updated = wijt + Bij@fijt + Aij@score  
    return fijt_updated

def update_non_playing_team(team_nr, ft_team_nr,psi,w_m ):  
    a1,a2,b1,b2,l3,delta = psi
    w_m_alpha, w_m_beta = w_m 
    alpha_mt, beta_mt = ft_team_nr

    alpha_mt_next = w_m_alpha + b1*alpha_mt
    beta_mt_next = w_m_beta + b2*beta_mt

    return np.array([alpha_mt_next, beta_mt_next]) #.reshape((2,1))
    


def update_round(ft,psi,w, t): 
    # if(t%100==0):
    #     print('updating f_t for round  ', t)
    #print('ft in UPDATE ROUND:', ft)
    all_teams = set(range(team_amount))  # num_teams is 33
    participants = participants_dict[t] #df[df["round_labels"] == t].iloc[0].participants
    #participants = set(participants)
    matches_this_round = matches_per_round[t]
    ft_next = ft
    # [  CASE 1  ]
    #new teams entering competition in this round
    
    new_teams = participants.intersection(team_tracker.unseen_teams)
    if new_teams:
        for team in new_teams:
            team_tracker.update_teams(team)
            #update_seen_teams(team)        #changes the set of seen_teams and unseen_teams    
            #set new_teams' strengths to 0
        ft = set_new_teams_parameters(new_teams,ft) #zet new_teams alpha en beta op 0 
            #calculate their w : gewoon op 0 laten staan

            #update their f_ij,t+1 accordingly -> gebeurt op zelfde manier als andere teams
    # [  CASE 2  ]
    #update teams not playing in this round BUT seen before (thus not in new_teams): 
    for match in matches_this_round:  # updates teams that actually play matches
        i, j = match #i and j are teams playing in this match
        selection = [i, j, i+team_amount, j+team_amount]
        fijt = ft[selection]
        wijt = w[selection]
        ft_next[selection] = update_fijt(fijt, i,j,psi, wijt,t)

    seen_teams_not_playing = all_teams.difference(participants).difference(team_tracker.unseen_teams)
    if seen_teams_not_playing:
        for team in seen_teams_not_playing:
            selection = [team, team+team_amount]
            #update alpha_team,t+1 = w_team + b1 * alpha_team,t
            #update beta_team,t+1 = w_team + b2 * beta_team,t
            ft_team = ft[selection]  # (alpha_mt, beta_mt)'
            w_m = w[selection] # (w_alpha_m, w_beta_m)'
            
            ft_next[selection] = update_non_playing_team(team, ft_team, psi, w_m )
    
    
    
    return ft_next 

def construct_w(f1, b1, b2):
    f1_values = get_f1().reshape((team_amount*2,1))
    ones_vec = np.ones((team_amount*2,1))
    diagonal_of_B = np.array(([b1]*team_amount,[b2]*team_amount)).reshape((team_amount*2,1))

    rhs = ones_vec - diagonal_of_B 
    result = np.multiply(f1_values, rhs)
    return result.reshape(66,)

 
alph, betas, alphas_before = [],[],[] 

rounds_in_first_year = np.max(df.round_labels.values[:380]) #in 38 rounds, 380 matches played in first year.
 
def update_all(f1, psi):   
    update_all.counter +=1 
    if (update_all.counter % 100 == 0):
        print(update_all.counter)
    #print("running UPDATE ALL")
     
    a1, a2, b1, b2, l3, delta = psi    
    w = construct_w(get_f1(),b1,b2)
    #make f1-vector the strength-vec of entire first year (first 38 rounds):
    first_year_strengths = np.multiply(np.ones((get_f1().shape[0], rounds_in_first_year)), get_f1().reshape((66, 1)))

    

    num_rounds = np.max(df["round_labels"].values)
    ft_total = np.zeros((get_f1().shape[0], num_rounds  ))
    ft_total[:, :38] = first_year_strengths  # [f1,f1,...,f1] for first 37 columns

    first_round = rounds_in_first_year 
    rounds = range(first_round, num_rounds)  
    for round in rounds:
        ft = 0 #placeholder
        #print('updating round: ', round)
        if round == first_round:
            ft = get_f1()
        else:
            ft = ft_total[:,round-1] 
             #constrains the alpha-values to make them sum to 1, avoiding overflows in likelihood.
            # alphas = ft[:team_amount]
            # alphas_before.append(alphas.sum())
            # alphas = constrain_alphas(alphas)
            # print(f"max of alphas = ", ft[:33].max())
            # print(f"max of betas = ", ft[33:].max())
            # betas.append(ft[33:].sum())
        #print('ft in UPDATE ALL', ft)
        ft_next = update_round(ft, psi, w, round) 
        ft_total[:,round] = ft_next
    #print("DONE WITH UPDATE ALL. Resetting Unseens and Seen Teams and Proceeding to likelihood calc:  ")
    team_tracker.reset()
    return ft_total
update_all.counter = 0 
 

#start calculating here...

def calc_round_likelihood(all_games_in_round, ft, delta, l3):
    #print('calculating likelihood for round ', all_games_in_round.iloc[0].round_labels)
    temp = np.zeros(all_games_in_round.shape[0])

    def game_likelihood(game, ft, delta, l3):
        team_amount = int(ft.shape[0]/2)
        home_team, away_team = int(game["HomeTeamCat"]), int(game["AwayTeamCat"])

        selection = [home_team, away_team, home_team +
                     team_amount, away_team+team_amount]
        alpha_i, alpha_j, beta_i, beta_j = ft[selection]

        x, y = game["y"]
        
        l1, l2 = bivariate_poisson.link_function(alpha_i, alpha_j, beta_i, beta_j, delta)
        game_ll = np.log(bivariate_poisson.pmf(x, y, l1, l2, l3))
        return game_ll
    temp = all_games_in_round.apply(game_likelihood, ft = ft, delta =delta, l3 = l3, axis = 1)
    return temp.sum(axis=0)


likelihood_list = []
#@CallCountDecorator
def total_log_like_score_driven(theta, *args): #  (f1, delta, l3, rounds_in_first_year)):
    total_log_like_score_driven.counter +=1
    if (total_log_like_score_driven.counter % 100 == 0 ):
        print("Amount of times calculating total log likelihood = ", total_log_like_score_driven.counter)
    #gets theta-parameters from the optimizer
    a1, a2, b1, b2, l3, delta = theta #these are the only 6 parameters that need estimations
    #f1 = sd.get_f1()  # is een dataFrame series.
     #f1, l3, delta, rounds_in_first_year = args #are estimated beforehand with Maher
    #using those parameters, gets ft_total from train_score_driven_model
    f_start, rounds_in_first_year = args
    f_start = get_f1()
    psi = (a1, a2, b1, b2, l3, delta)
    ft_total = update_all(f_start.copy(), psi)
    #print('ft_total succesfully calculated')
    #print("Used params   a1, a2, b1, b2, l3, delta = ", psi)
    #print("ft_total has shape: ", ft_total.shape)
    #print('FT_TOTAL: ', ft_total)
    #calculates the total log likelihood and returns it
    #print(f"f1 VECTOR: ", f_start)
    #optimizer optimizes, returns optimal estimated parameter-vector.

    first_round_index = rounds_in_first_year 
    last_round = ft_total.shape[1]

    total_likelihood = 0
    #print('total_likelihood is starting iteration over rounds: ')
    for round in range(first_round_index, last_round):
        #retrieve (x,y) score for this round from df
        #retrieve related strength-vector f_round from ft_total
        #feed to calc_round_likelihood
        all_games_in_round = df[df.round_labels == round]
        f_t = ft_total[:, round]
        round_likelihood = calc_round_likelihood(
            all_games_in_round, f_t, delta, l3)
        total_likelihood += round_likelihood

    likelihood_list.append((-1*total_likelihood,(a1,a2,b1,b2, l3,delta),construct_w(get_f1(),b1,b2), ft_total))

    minimize=True
    if minimize == True:
        # if optimizer uses minimisation in stead of maximisation.
        #print(f"total log-likelihood: {-1*total_likelihood}")
        return -1*total_likelihood
    print(f"total log-likelihood: {total_likelihood}")
    return total_likelihood

total_log_like_score_driven.counter = 0


def callback_func(xk, convergence):
    print("Data passed to callback: " + str(xk))
    print("Optimizer callback count: " + str(callback_func.counter))
    callback_func.counter += 1



callback_func.counter = 0


def optimizer():
    #theta contains the paramters to optimize: a1,a2,b1,b2
    #args takes additional parameters that won't be optimized:
    #args = (f1, delta_1, lambda_3, rounds_in_first_year)
    f_start = get_f1()
    print('succesfully retrieved f1 for optimizer')
    l3_start, delta_start = lambda_3, delta_1
    arguments = (f_start, rounds_in_first_year)
    # a1,a2,b1,b2, l3, delta = theta
    theta_ini = [-0.17, 0.165, -0.97, 0.98, 0.02, 0.26]
    #min_bound, max_bound = -0.99, 0.99
    a_bounds = [-4, 4]
    b_bounds = [-4,4]
    l3_bounds = [0, 1]
    delta_bounds = [-1, 1]
    theta_bounds = np.array([a_bounds, a_bounds, b_bounds, b_bounds, l3_bounds, delta_bounds])
    print('starting optimizer: ...')
    max_iterations = 500
    #results = scipy.optimize.dual_annealing(total_log_like_score_driven,  args=arguments, no_local_search = False,  x0=theta_ini, bounds=theta_bounds, maxiter=max_iterations)#, callback=callback_func) #, options={'disp': True})  # , options={'xatol': 1e-8, 'disp': True})
 
    #results = scipy.optimize.differential_evolution(total_log_like_score_driven, bounds=theta_bounds, args=arguments, strategy='best1bin', maxiter=max_iterations, popsize=15, tol=0.01, mutation=(
        #0.5, 1), recombination=0.7, seed=None, callback=callback_func, disp=True, polish=True, init='latinhypercube', atol=0, updating='immediate', workers=1, constraints=())

    results = scipy.optimize.minimize(total_log_like_score_driven, theta_ini, args=arguments,
                                        method='SLSQP',
                                       constraints=(),
                                       bounds=theta_bounds)
    return results
results = optimizer()

print("DONE")
print(results)

print('pickling and saving results and likelihood list as results.pickle and likelihood_list.pickle in ', os.getcwd())
# with open('results.pkl', 'wb') as f:
#     pkl.dump(results, f)

# with open('likelihood_list.pkl', 'wb') as f:
#     pkl.dump(likelihood_list, f)

toc = time.perf_counter()

likelihoods = [i[0] for i in likelihood_list]
psis = [i[1] for i in likelihood_list]
ft_totals = [i[3] for i in likelihood_list]

#laat het verloop van a1,a2,b1,b2 estimates zien
dfpsis = pd.DataFrame(np.array(psis))
dfpsis.columns = ["a1", "a2", "b1", "b2" ,"lambda3", "delta"]
dfpsis.iloc[-200:,:].plot()


#check Barcelona - Real Madrid strengths-verloop: 
f  =  ft_totals 
lastf = f[-1]
last = pd.DataFrame(lastf)
barca = inv_mapping.get('Barcelona')     #=5
real = inv_mapping.get('Real Madrid')   #= 21 
barca_madr = last.iloc[[barca, real]]    
barca_madr.T.plot()

final_lambda, final_delta  = psis[-1][-2], psis[-1][-1]

def add_strengths(round_group):
    round_group.ai = round_group.round_labels
    round_strengths = lastf[:,round]


def add_strength_columns_to_df(df, lastf, final_lambda, final_delta):
    df["ai aj bi bj lambda delta".split()] = 'nan'
    team_amount = lastf.shape[0]/2
    for round in set(df.round_labels):
        for match_row in df[df.round_labels == round].values:
            home, away = match_row.HomeTeamCat, match_row.AwayTeamCat 
            #match_row[]lastf[[home, away, home+team_amount, away+team_amount],round]


def get_ai_aj_etc(row, lastf):
    home_team_cat, away_team_cat,  round = row.AwayTeamCat, row.HomeTeamCat, row.round_labels
    team_amount = lastf.shape[0]/2
    selection = [home_team_cat, away_team_cat, home_team_cat+team_amount, away_team_cat + team_amount]
    selection = [int(i) for i in selection]
    print(f"selection {selection}")
    ai,aj,bi,bj = lastf[selection,round-1]
    print(ai, aj, bi, bj )
    return np.array((ai, aj, bi, bj))
def create_strength_columns(df, lastf, final_lambda, final_delta):
    ai, aj, bi, bj, fin_lamn, fin_delt = [],[],[],[], [], []
    for i in df.index:
        a,b,c,d = get_ai_aj_etc(df.iloc[i], lastf)
        ai.append(a)
        aj.append(b)
        bi.append(c)
        bj.append(d)
        fin_lamn.append(final_lambda)
        fin_delt.append(final_delta)
    return ai, aj, bi, bj, fin_lamn, fin_delt


df["ai aj bi bj lambda delta".split()] = 'nan'
ai, aj, bi, bj, lambd, delt  = create_strength_columns(df, lastf, final_lambda, final_delta)
df['ai'], df['aj'], df['bi'], df['bj'], df['lambda'], df['delta'] = ai, aj, bi, bj, lambd, delt


with open('df_trained.pkl', 'wb') as f:
     pkl.dump(df, f)



print(f"took {toc-tic} seconds to run")
# %%
