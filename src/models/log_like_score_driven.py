import numpy as np 
import pandas as pd 
import bivariate_poisson
import train_score_driven_model as sd
from scipy import optimize
from scipy import stats
import scipy





def calc_round_likelihood(all_games_in_round, ft, delta, l3 ):
    total_round_likelihood = 0        
    for game in all_games_in_round: 
            team_amount = ft.shape[0]/2
            home_team, away_team = game["HomeTeamCat"], game["AwayTeamCat"]
            
            selection = [home_team, away_team, home_team+team_amount, away_team+team_amount]
            alpha_i, alpha_j, beta_i, beta_j = ft[selection]
            
            x, y = game["y"]
            l1 = np.exp(alpha_i - beta_j + delta)
            l2 = np.exp(alpha_j - beta_i)

            total_round_likelihood += np.log(bivariate_poisson.pmf(x, y, l1, l2, l3))
            
    return total_round_likelihood

    
def total_log_like_score_driven(theta, df, rounds_in_first_year, minimize = True):
    #gets theta-parameters from the optimizer
    a1, a2, b1, b2, l3, delta = theta
    f1 = sd.get_f1() #is een dataFrame series.
    

    #using those parameters, gets ft_total from train_score_driven_model
    ft_total = score_driver.update_all(f1,theta)
    #calculates the total log likelihood and returns it
    
    #optimizer optimizes, returns optimal estimated parameter-vector.
    
    

    first_round_index = 0 
    last_round = ft_total.shape[1]
    
    
    total_likelihood = 0 

    for round in range(first_round_index,last_round): 
        #retrieve (x,y) score for this round from df
        #retrieve related strength-vector f_round from ft_total
        #feed to calc_round_likelihood
        all_games_in_round = df[df.round.labels == round]
        f_t = ft_total[:,round]
        round_likelihood = calc_round_likelihood(all_games_in_round, f_t, delta, l3)
        total_likelihood += round_likelihood

    if minimize == True:
        return -1*total_likelihood #if optimizer uses minimisation in stead of maximisation.
    return total_likelihood

def optimize():
    #theta contains the paramters to optimize: a1,a2,b1,b2
    #args takes additional parameters that won't be optimized: 
    #args = (delta, l3,w )
    theta_ini = 0
    results = scipy.optimize.minimize(total_log_like_score_driven, theta_ini, args=(first_year),
                                      options=options,
                                      method='SLSQP',
                                      constraints=cons,
                                      bounds=boundaries)
