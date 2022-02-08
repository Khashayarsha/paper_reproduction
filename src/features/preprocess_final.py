# %%
# %reset - f
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

# purpose of this module is to get raw data, and process it such that
# maher_initialization.py can initialize the first power-vector

#in:   raw data.csv from TimeSeriesLab
#out:  prepped_data.pkl for use in maher_initialization
#      saves it in ../../data/interim  folder


data_path = "../../data/raw"
print("Expected directory containing data = ", data_path)
os.chdir(data_path)

# TimeSeriesLab -> download -> first = 1999, last =2008
in_sample = data_path+"\SpanishPrimera99-08.csv"
# TSL ->dl -> first = 2009, last = 2015
out_sample = data_path+"\SpanishPrimera09-15.csv"

df = pd.read_csv(in_sample)
df_out = pd.read_csv(out_sample)

relevant_columns = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']

df, df_out = df[relevant_columns], df_out[relevant_columns]
N_sample = df.shape[0]
N_outsample = df_out.shape[0]

home_teams = df["HomeTeam"]
away_teams = df["AwayTeam"]
df["HomeTeamCat"] = pd.Categorical(home_teams).codes
df["AwayTeamCat"] = pd.Categorical(away_teams).codes

temp = df[["HomeTeam", "HomeTeamCat"]].values
temp2 = set((j, i) for [i, j] in temp)
mapping = dict(temp2)

def get_first_year(data, test =True):
    #returns the part of the DataFrame which encompasses the first year of matches
    #for testing purposes, it's hardcoded to the first 380 matches. 
    if test == True: 
        first_year = data.iloc[:380]   
        return first_year

first_year = get_first_year(df, test = True)






def get_participants_per_round(data, round):
    # based on round label:
    # return the set of all teams that played in this round
    home = set(data[data["round_labels"] == round]["HomeTeamCat"])
    away = set(data[data["round_labels"] == round]["AwayTeamCat"])
    participants = home.union(away)
    return list(participants)


def get_participants(data):
    round_labels = data["round_labels"]
    result = []
    for i in round_labels:
        if i % 20:
            print("adding participants", i)
        sub = get_participants_per_round(data, i)
        result.append(sub)
    data["participants"] = result
    print("Added participants")
    return data


def construct_round_labels(tau, data):
    diff = pd.DataFrame(tau).diff(1)
    diff.iloc[0] = tau[0]
    diff = list(i for [i] in diff.values)
    diff.append(len(data)-tau[-1])

    result = []
    for (i, j) in enumerate(diff):
        temp = [i]*int(j)
        result.append(temp)
    return [item + 1 for sublist in result for item in sublist]


def add_rounds(data):
    # input: match-participation data (HomeTeamCat, AwayTeamCat)
    # return: a series of indices tau that define start of new round
    # tau_0 = 0

    seen = set()
    home = data["HomeTeamCat"]
    away = data["AwayTeamCat"]

    tau = []
    participants = []
    for i in range(len(data)):
        if home.iloc[i] in seen or away.iloc[i] in seen:
            print("seen before: ")
            participants.append([j for j in seen])
            seen = set()
            tau.append(i)

        else:
            seen.add(home.iloc[i])
            seen.add(away.iloc[i])

    # creates list of round-labels
    round_labels = construct_round_labels(tau, data)
    data["round_labels"] = round_labels
    # data"participants"] = participants[
    data = get_participants(data)
   # data["participants"] =
    return data
    # %%


def num_unique_teams(data, which="False"):
    if which == False:
        return len(list(set(data[["HomeTeamCat", "AwayTeamCat"]].values.flatten())))
    else:
        number = len(
            list(set(data[["HomeTeamCat", "AwayTeamCat"]].values.flatten())))
        which_teams = set(
            data[["HomeTeamCat", "AwayTeamCat"]].values.flatten())
        return number, which_teams

def construct_y(data):
    data["y"] = [(i, j) for i, j in df[["FTHG", "FTAG"]].values]
    return data


# returns set of absentees (teams that didn't play in first year)
def get_absentees(data, model="1"):
    if model == "1":
        size = 2 * num_unique_teams(df)
        num_first_year, set_first_year = num_unique_teams(
            first_year, which=True)  # teams that played first year
        _, set_all_teams = num_unique_teams(df, which=True)
        # teams that didn't play in first year, but did after
        #absentees = which_first_year.difference(all_teams)
        absentees = set_all_teams.difference(set_first_year)
    return absentees


df = add_rounds(df)
df = construct_y(df)
absentees = np.zeros_like(df['participants'])
absentees[0] = get_absentees(df)
df['absentees'] = absentees
first_year = df.iloc[:380]
current_t = 1  # counter

mappert = np.zeros_like(df['participants'])
mappert[0] = mapping 
df['mapping'] = mappert

import time as time
def timestamp_name(name):
    #adds timestamp to name
    timestamp = str(time.localtime()[1:-3])[1:-1].replace(', ', '_')
    name += '_'+timestamp
    return name

 

name = "prepped_data" 
name = timestamp_name(name) + '.pkl'
df.to_pickle(name)
df.to_csv(name+'.csv')
print("done with data prep")
print('test if pickled correctly:')
data = pd.read_pickle(name)

# %%

