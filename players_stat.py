#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 03:49:17 2022

@author: k1
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

def get_best_squad(formation, team):
    store = []
    
    # iterate through all positions in the input formation and get players with highest overall respective to the position
    for i in formation:
        store.append([
            i,
            team.loc[[team[team['Position'] == i]['overall'].idxmax()]]['short_name'].to_string(index=False),
            team[team['Position'] == i]['overall'].max(),
            team.loc[[team[team['Position'] == i]['overall'].idxmax()]]['age'].to_string(index=False),
            team.loc[[team[team['Position'] == i]['overall'].idxmax()]]['club_name'].to_string(index=False),
            team.loc[[team[team['Position'] == i]['overall'].idxmax()]]['value_eur'].to_string(index=False),
            team.loc[[team[team['Position'] == i]['overall'].idxmax()]]['wage_eur'].to_string(index=False)
        ])
        team.drop(team[team['Position'] == i]['overall'].idxmax(), 
                         inplace=True)
    # return store with only necessary columns
    x = team['overall'].sum()
    return x, pd.DataFrame(np.array(store).reshape(11,7), 
                        columns = ['Position', 'short_name', 'overall', 'age', 'club_name', 'value_eur', 'wage_eur']).to_string(index = False)
def best_lineup_func(team, squads):
    s_all = []
    squad_all = []
    team_linup_all = []
    for i,v in enumerate(squads):
        s, team_linup = get_best_squad(v, team)
        squad_all.append(squad_names[i])
        s_all.append(s)
        team_linup_all.append(team_linup)
    s_best = max(s_all)
    squad_best = squad_all[s_all.index(s_best)]
    linup_best = team_linup_all[s_all.index(s_best)]
    summery_all = [squad_all, s_all, squad_best, team_linup_all] 
    summery_best = [squad_best, linup_best]
    return summery_all, summery_best


FIFA22 = pd.read_csv('./players_22.csv')

interesting_columns = ['short_name', 'age', 'nationality_name', 'overall', 'potential', 'club_name', 'value_eur', 'wage_eur', 'player_positions']
FIFA22 = pd.DataFrame(FIFA22, columns=interesting_columns)

list_2022 = ['Qatar', 'Germany', 'Denmark', 'Brazil', 'France', 'Belgium', 'Croatia', 'Spain', 'Serbia', 'England', 'Switzerland', 'Netherlands', 'Argentina', 'IR Iran', 'Korea Republic', 'Japan', 'Saudi Arabia', 'Ecuador', 'Uruguay', 'Canada', 'Ghana', 'Senegal', 'Portugal', 'Poland', 'Tunisia', 'Morocco', 'Cameroon', 'USA', 'Mexico', 'Wales', 'Australia', 'Costa Rica']
FIFA22['Position'] = FIFA22['player_positions'].str.split(",").str[0]
FIFA22 = FIFA22[["short_name", "age", "nationality_name", 'overall', 'potential', "club_name", "Position", "value_eur", "wage_eur"]]


FIFA22 = FIFA22[(FIFA22["nationality_name"].apply(lambda x: x in list_2022))]
Denmark_team = FIFA22[FIFA22["nationality_name"]=='Denmark']

squad_433 = ['GK', 'RB', 'CB', 'CB', 'LB', 'CDM', 'CM', 'CAM', 'RW', 'ST', 'LW']
squad_442 = ['GK', 'RB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'LM', 'ST', 'ST']
squad_4231 = ['GK', 'RB', 'CB', 'CB', 'LB', 'CDM', 'CDM', 'CAM', 'CAM', 'CAM', 'ST']
squad_343= ['GK', 'CB', 'CB', 'CB', 'RWB', 'CDM', 'CDM', 'LWB', 'RW', 'CF', 'LW']

squads = [squad_433, squad_442, squad_4231,squad_343]
squad_names = ['433','442','4231','343']

summery_all, summery_best = best_lineup_func(Denmark_team, squads)


