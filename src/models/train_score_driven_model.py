# %%
# %reset -f
import pandas as pd
import numpy as np
import bivariate_poisson
import os
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

unseen_teams = df['absentees'].iloc[0]  #set of teams not playing in first round
seen_teams = set(df.iloc[0].participants) #set of teams that played in first round (from which Maher init calculated)


temp = [df[df["round_labels"] == t][["HomeTeamCat", "AwayTeamCat"]].values for t in set(df["round_labels"])]
# print("eval start")  #geen idee wat dit doet :') 05-01-2021
# df['participants'] = df.participants.apply(eval)
# print("eval end")
matches_per_round = [[(i, j) for i, j in round] for round in temp]
#turn into a dict for performance and correct indexing:
matches_per_round = {i+1: matches_per_round[i] for i in range(len(matches_per_round))} 


seen_teams = set()
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
    matches_played = matches_per_round(round)
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

def update_seen_teams(seen_team):
    seen_teams.add(seen_team)
    unseen_teams.remove(seen_team)


def update_non_playing_team(i, t, psi, ft):
    """Updates the team strengths and defences of teams
     that did not play in a specific round """

    # how to handle the case where a previously unseen team plays against a
    # previously seen team?
    

    #w_i, B_i = psi[select w and B]
    f_it_next = 0
    f_it_next += w_ij + B_ij*f_ijt + A_ij * score

    return 0


def update_fijt(i, j, psi, t, f_t,  in_round):
    """ updates strengths and defences of teams i and  j 
    according to the updating rule per match played in a round."""

    # f_ijt+1 = w_ij + B_ij * f_ijt + A_ij * s_ijt
    a1, a2, b1, b2, l3, delta = psi

    score = 0

    M_ij = select_dict[(i, j)]

    f_ijt = M_ij@f_t
    alpha_i, alpha_j, beta_i, beta_j = f_ijt
    l1, l2 = link_function(alpha_i, beta_j, delta, alpha_j, beta_i)

    if in_round:
        x = get_home_goals(i, j, t)
        y = get_away_goals(i, j, t)
        score = bivariate_poisson.score(x, y, l1, l2, l3)
    else:
        score = 0

    w_ij = M_ij@w
    #A_ij = np.multiply(np.multiply(M_ij, A), M_ij.T)
    #B_ij = np.multiply(np.multiply(M_ij, B), M_ij.T)

    B_ij = np.diag([b1, b1, b2, b2])  # auto-regressive part
    A_ij = np.diag([a1, a1, a2, a2])  # score-updating part

    f_ijt_next = 0
    f_ijt_next += w_ij + B_ij*f_ijt + A_ij * score

    return f_ijt_next            # (a_it_next, a_jt_next, b_it_next, b_jt_next)

def select_rows_to_update_played(i,j):
    #returns numpy indices of the rows to be updated based on team_numbers i and j 
    # selects rows to update ait, ajt, bit, bjt respectively
    return [i, j, i+team_amount, j+team_amount] 
def select_rows_to_update_absent(i):
    # selects correct rows for updating a_it and b_it respectively
    return 0 

def update_round(t, psi, ft, num_teams =33):
    """gets psi and current strengths and defences (ft) and calculates
    according to score-update rule from Lit 2017 the next f_t+1 and returns this.

    It does this by handling all occuring matches in a round using update_fijt()
    and handling all teams that did not play a match in a round using update_non_playing()"""
    #f_t = ...
    #teams_in_round = set()
    all_teams = set(range(num_teams)) #num_teams is 33
    participants = df[df["round_labels"] == t].iloc[0].participants    
    teams_not_in_round = all_teams.difference(participants)         
    matches_played = matches_per_round[t]

    for match in matches_played:  # updates teams that actually play matches
        i,j = match
        M_ij = select_dict[(i, j)]
        ait, ajt, bit, bjt = update_fijt( i, j, psi,t, f_t, in_round=True)  # ait_next, bit_next
        selection = select_rows_to_update_played(i, j)
        updated = np.array([ait, ajt, bit, bjt])
        f_t[selection] = updated
    print(f"round {t}, done updating teams that have played matches")
    for team in teams_not_in_round:
        # ait, bit should be called ait_next, bit_next
        ait_next, bit_next = update_non_playing_team(i, f_t, psi)
        selection = select_rows_to_update_absent(i)
        updated = np.array([ait_next, bit_next])
        f_t[selection] = updated
    print(f"round {t}, done updating teams that have NOT played matches")
    f_t_next = f_t  # just for naming
    return f_t_next

def set_new_teams_parameters(new_teams, ft): 
    #initializes strenghts and defences to 0 for newly-appeared teams
    new_teams = list(new_teams)
    alpha_locs = new_teams
    beta_locs  = np.array(new_teams) + team_amount

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

def update_non_playing_team(team_nr, ft_team_nr,b1,b2,w_m ):
    w_m_alpha, w_m_beta = w_m 
    alpha_mt, beta_mt = ft_team_nr

    alpha_mt_next = w_m_alpha + b1*alpha_mt
    beta_mt_next = w_m_beta + b2*beta_mt

    return np.array([alpha_mt_next, beta_mt_next]).reshape((2,1))
    


def update_round(ft,psi,w, t):
    all_teams = set(range(team_amount))  # num_teams is 33
    participants = df[df["round_labels"] == t].iloc[0].participants
    matches_this_round = matches_per_round[t]
    ft_next = ft
    # [  CASE 1  ]
    #new teams entering competition in this round
    new_teams = participants.intersection(unseen_teams)
    if new_teams:
        for team in new_teams:
            update_seen_teams(team)        #changes the set of seen_teams and unseen_teams    
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

    seen_teams_not_playing = all_teams.difference(participants).difference(unseen_teams)
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


def log_likelihood_score_driven(theta, data, minimize=True):
    # adjust to time-varying model: 

    n = team_amount
    # print(x)

    a1, a2, b1, b2, w, lambda_3, delta, f1 = theta
    a1, a2, b1, b2, lambda_3, delta = psi
    alphas = f_t[:n]
    betas = f_t[n:2*n]

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
        l3 = lambda_3
        total += np.log(bivariate_poisson.pmf(x, y, l1, l2, l3))
    #total += 0.1 * np.sum(alphas)
    if minimize == True:
        return -1*total
    return total


rounds_in_first_year = np.max(df.round_labels.values[:380]) #in 38 rounds, 380 matches played in first year.

def update_all(f1, psi):
    a1, a2, b1, b2, lambda3, delta = psi    
    w = construct_w(f1,b1,b2)

    num_rounds = np.max(df["round_labels"].values)
    ft_total = np.zeros((f1.shape[0], num_rounds  ))
    ft_total[:,0] = f1.values   #Maher-estimated vector is f_0 
    first_round = rounds_in_first_year + 1
    rounds = range(first_round, num_rounds) 
    for round in rounds:

        ft = ft_total[:,round-1] 
        ft_next = update_round(round, psi, ft)
        ft_total[:,round] = ft_next

    return ft_total

# def construct_omega(f1, B):
#     #following Koopman/Lit omega is constructed by  w = f1.pointwise(    ( 1-Diag(B)  )    )
#     diag_B = 1
#     return 0 

#start calculating here...



print("DONE")



# %%
