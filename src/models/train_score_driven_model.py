# %%
# %reset -f
import pandas as pd
import numpy as np
import bivariate_poisson
import os
from scipy import optimize
from scipy.optimize import minimize
from scipy import stats
import scipy
from TeamTracker import TeamTracker

data_path = r"../../data/interim"
print("Expected directory containing data = ", data_path)
os.chdir(data_path)

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

use_small_data = True
if use_small_data == True: 
    df = df.iloc[:700]



#import preprocess
team_amount = 33     #turn into a   function or do this in pre-process etc
f1_size = (team_amount * 2) #+2    f_t just consists of the time-varying team atk and def str's. 
                            # DELTA AND GAMMA ARE NOT TIME VARYING IN THIS MODEL.
# psi1 = pd.read_csv("f1_maher_init2.csv")
# print("TEST")
# df = pd.read_csv("processed_data2.csv")
f1 = df['f1'][:f1_size]
lambda_3 = f1.values[-2]  # noemde dit eerst gamma_1, maar is bs
delta_1 = f1.values[-1]
mapping = df['mapping'].iloc[0] #dictinary that maps numbers to teams.

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

def selec_mat(combination, N=33):
    # returns the selection-matrix that selects (a_i, a_j, B_i, B_j)' when
    # post-multiplied with f_t
    # f_ijt = selec_mat(i,j,N) * f_t
    i, j = combination
    M = np.zeros((4, 2*N))
    M[:, i] = np.array((1, 0, 0, 0))
    M[:, j] = np.array((0, 1, 0, 0))
    M[:, N+i] = np.array((0, 0, 1, 0))
    M[:, N+j] = np.array((0, 0, 0, 1))

    return M


def construct_selection_mat_dict(n):
    """Returns a dictionary with keys (home, away)
     and values the corresponding selection matrix M_ij. 
     As to not construct them ad-hoc while calculating likelihood """
    combinations = []
    for i in range(n):
        for j in range(n):
            if i != j:
                combinations.append((i, j))
    selection_dictionary = dict(
        zip(combinations, map(selec_mat, combinations)))
    return selection_dictionary


select_dict = construct_selection_mat_dict(team_amount)   #get this from helper-code


def link_function(alpha_i, beta_j, delta, alpha_j, beta_i):
    l1_ij = np.exp(delta + alpha_i - beta_j)
    l2_ij = np.exp(alpha_j - beta_i)
    return l1_ij, l2_ij


def get_goals(i, j, t):
    """ Returns home_goals, away_goals given two teams (integer category).
    Returns a string if there exists no match between home and away in round t """

    home, away, round = i, j, t

    # verify team i plays team j in round t
    matches_played = matches_per_round.get(round)
    match_exists = (home,away) in matches_played
    if not match_exists:
        print(f"Match {(home,away)} in round {round} does not exist ")
        return "get_goals method failed because MATCH DOESN'T EXIST"
    else:
        match = df[(df["round_labels"] == round) & (df["HomeTeamCat"] == home) & (df["AwayTeamCat"] == away)]
        home_score = match.FTHG.values[0]
        away_score = match.FTAG.values[0]
    return home_score, away_score

def get_home_goals(i,j,t):
    home_goals, away_goals = get_goals(i,j,t)
    return home_goals

def get_away_goals(i, j, t):
    home_goals, away_goals = get_goals(i, j, t)
    return away_goals

# def update_seen_teams(seen_team):
#     seen_teams.add(seen_team)
#     unseen_teams.remove(seen_team)

# def reset_unseen_and_seen_teams():
#     return unseen_teams_init, seen_teams_init







def set_new_teams_parameters(new_teams, ft): 
    #initializes strenghts and defences to 0 for newly-appeared teams
    new_teams = list(new_teams)
    alpha_locs = new_teams
    beta_locs  = np.array(new_teams) + team_amount

    #print('ft in set_new_teams_parameters: ', ft)
    #print('alpha locs: ', alpha_locs)
    #print('beta_locs: ', beta_locs)

    ft[alpha_locs] = 0
    ft[beta_locs] = 0
    return ft
def get_f1():
    return f1
def update_fijt(fijt,i,j,psi,wijt,t):
    a1, a2, b1, b2, l3, delta = psi
    alpha_i, alpha_j, beta_i, beta_j = fijt
    score = 0
    fijt_updated = fijt 
    
    x = get_home_goals(i, j, t)
    y = get_away_goals(i, j, t)
    l3 = lambda_3
    
    score = bivariate_poisson.score(fijt,x, y, l3, delta) #must return a 4x1 vector
    

    Bij = np.diag([b1,b1,b2,b2])
    Aij = np.diag([a1,a1,a2,a2])

    fijt_updated = wijt + Bij@fijt + Aij@score  
    return fijt_updated

def update_non_playing_team(team_nr, ft_team_nr,psi,w_m ):  
    a1,a2,b1,b2,_l3,_delta = psi
    w_m_alpha, w_m_beta = w_m 
    alpha_mt, beta_mt = ft_team_nr

    alpha_mt_next = w_m_alpha + b1*alpha_mt
    beta_mt_next = w_m_beta + b2*beta_mt

    return np.array([alpha_mt_next, beta_mt_next]) #.reshape((2,1))
    


def update_round(ft,psi,w, t): 
    if(t%100==0):
        print('updating f_t for round  ', t)
    #print('ft in UPDATE ROUND:', ft)
    all_teams = set(range(team_amount))  # num_teams is 33
    participants = df[df["round_labels"] == t].iloc[0].participants
    participants = set(participants)
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
    f1_values = f1.values.reshape((team_amount*2,1))
    ones_vec = np.ones((team_amount*2,1))
    diagonal_of_B = np.array(([b1]*team_amount,[b2]*team_amount)).reshape((team_amount*2,1))

    rhs = ones_vec - diagonal_of_B 
    result = np.multiply(f1_values, rhs)
    return result.reshape(66,)

# def construct_w(f1, B):
#     f1_values = f1.values.reshape((1,66)) 
#     ones_vec = np.ones((1,B.shape[0]))
#     diagonal_B = np.diag(B) 
#     rhs = ones_vec - diagonal_B
#     result = np.multiply(f1_values, rhs) #point-wise multiplication of f1*(1-diag(B))
#     return result




rounds_in_first_year = np.max(df.round_labels.values[:380]) #in 38 rounds, 380 matches played in first year.
 
def update_all(f1, psi):   
    print("running UPDATE ALL")
     
    a1, a2, b1, b2, lambda3, delta = psi    
    w = construct_w(f1,b1,b2)
    #make f1-vector the strength-vec of entire first year (first 38 rounds):
    first_year_strengths = np.multiply(np.ones((f1.shape[0], rounds_in_first_year)), f1.values.reshape((66, 1)))

    

    num_rounds = np.max(df["round_labels"].values)
    ft_total = np.zeros((f1.shape[0], num_rounds  ))
    ft_total[:, :38] = first_year_strengths  # [f1,f1,...,f1] for first 37 columns

    first_round = rounds_in_first_year 
    rounds = range(first_round, num_rounds)  
    for round in rounds:
        ft = 0 #placeholder
        print('updating round: ', round)
        if round == first_round:
            ft = f1.values
        else:
            ft = ft_total[:,round-1] 
        #print('ft in UPDATE ALL', ft)
        ft_next = update_round(ft, psi, w, round) 
        ft_total[:,round] = ft_next
    print("DONE WITH UPDATE ALL. Resetting Unseens and Seen Teams and Proceeding to likelihood calc:  ")
    team_tracker.reset()
    return ft_total

# def construct_omega(f1, B):
#     #following Koopman/Lit omega is constructed by  w = f1.pointwise(    ( 1-Diag(B)  )    )
#     diag_B = 1
#     return 0 

#start calculating here...

def calc_round_likelihood(all_games_in_round, ft, delta, l3):
    print('calculating likelihood for round ', all_games_in_round.iloc[0].round_labels)
    temp = np.zeros(all_games_in_round.shape[0])

    def game_likelihood(game, ft, delta, l3):
        team_amount = int(ft.shape[0]/2)
        home_team, away_team = int(game["HomeTeamCat"]), int(game["AwayTeamCat"])

        selection = [home_team, away_team, home_team +
                     team_amount, away_team+team_amount]
        alpha_i, alpha_j, beta_i, beta_j = ft[selection]

        x, y = game["y"]
        l1 = np.exp(alpha_i - beta_j + delta)
        l2 = np.exp(alpha_j - beta_i)

        game_ll = np.log(bivariate_poisson.pmf(x, y, l1, l2, l3))
        return game_ll
    temp = all_games_in_round.apply(game_likelihood, ft = ft, delta =delta, l3 = l3, axis = 1)
    return temp.sum(axis=0)


likelihood_list = []
def total_log_like_score_driven(theta,  f1, delta, l3, rounds_in_first_year):
    #gets theta-parameters from the optimizer
    a1, a2, b1, b2 = theta #these are the only 4 parameters that need estimations
    #f1 = sd.get_f1()  # is een dataFrame series.
     #f1, l3, delta, rounds_in_first_year = args #are estimated beforehand with Maher
    #using those parameters, gets ft_total from train_score_driven_model
    psi = (a1, a2, b1, b2, l3, delta)
    ft_total = update_all(f1, psi)
    print('ft_total succesfully calculated')
    #print("ft_total has shape: ", ft_total.shape)
    #print('FT_TOTAL: ', ft_total)
    #calculates the total log likelihood and returns it

    #optimizer optimizes, returns optimal estimated parameter-vector.

    first_round_index = rounds_in_first_year 
    last_round = ft_total.shape[1]

    total_likelihood = 0
    print('total_likelihood is starting iteration over rounds: ')
    for round in range(first_round_index, last_round):
        #retrieve (x,y) score for this round from df
        #retrieve related strength-vector f_round from ft_total
        #feed to calc_round_likelihood
        all_games_in_round = df[df.round_labels == round]
        f_t = ft_total[:, round]
        round_likelihood = calc_round_likelihood(
            all_games_in_round, f_t, delta, l3)
        total_likelihood += round_likelihood

    likelihood_list.append((-1*total_likelihood,(a1,a2,b1,b2),construct_w(f1,b1,b2), ft_total))

    minimize=True
    if minimize == True:
        # if optimizer uses minimisation in stead of maximisation.
        return -1*total_likelihood
    return total_likelihood


def optimizer():
    #theta contains the paramters to optimize: a1,a2,b1,b2
    #args takes additional parameters that won't be optimized:
    #args = (f1, delta_1, lambda_3, rounds_in_first_year)
    f1 = get_f1()
    print('succesfully retrieved f1 for optimizer')
    arguments = (f1, delta_1, lambda_3, rounds_in_first_year)
    # a1,a2,b1,b2 = theta
    theta_ini = [0.1, 0.1, 0.1, 0.1]
    print('starting optimizer: ...')
    results = scipy.optimize.minimize(total_log_like_score_driven, theta_ini, args =arguments, method='nelder-mead',options={'xatol': 1e-8, 'disp': True})
    
    
    # results = scipy.optimize.minimize(total_log_like_score_driven, theta_ini, args=arguments,
    #                                   options=options,
    #                                   method='SLSQP',
    #                                   constraints=cons,
    #                                   bounds=boundaries)
    return results
results = optimizer()
print("DONE")



# %%
