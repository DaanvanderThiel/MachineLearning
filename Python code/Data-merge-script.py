# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#To run from this place, press F9
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# (!!!) Set your own directory (!!!)
os.chdir('/Users/dorienVU/Documents/GitHub/MachineLearning/Python code')

# Load files
combats = pd.read_csv('../Data/Original data/combats.csv')
pokemon = pd.read_csv('../Data/Original data/pokemon.csv')

#Set the index of the pokemon table equal to the ID of the pokemon
pokemon = pokemon.set_index('#')
                            
# convert Legendary to binary, lower case types
pokemon['Legendary'] = pokemon['Legendary'].apply(lambda x: 1 if x else 0)
pokemon['Type 1'] = pokemon['Type 1'].str.lower()
pokemon['Type 2'] = pokemon['Type 2'].str.lower()

# Add stats of first and second pokemon
columns = pokemon.columns.values
for i in columns:
    combats['{}_{}'.format(i, '1')] = np.nan
    
f = lambda i, j: pokemon[i].loc[j]

for i in combats.columns[3:]:
    combats[i] = [f(i.rsplit('_')[0], j) for j in combats['First_pokemon']]

for i in columns:
    combats['{}_{}'.format(i, '2')] = np.nan

for i in combats.columns[14:]:
    combats[i] = [f(i.rsplit('_')[0], j) for j in combats['Second_pokemon']]
    
# Add correct multipliers
multipliers = pd.read_csv('../Data/Original data/type-chart.csv')

def get_types(pokemon_id):
    return pd.Series(pokemon[['Type 1', 'Type 2']].loc[pokemon_id]).dropna()

def find_multipliers(pokemon1_id, pokemon2_id):
    p1_types, p2_types = get_types(pokemon1_id), get_types(pokemon2_id)
    
    # select columns of interest
    defense_columns = pd.Series(['defense-type1', 'defense-type2'])
    multiplier_p1 = multipliers[defense_columns.append(p1_types)]
    multiplier_p2 = multipliers[defense_columns.append(p2_types)]
    
    # select defense-type1
    multiplier_p1 = multiplier_p1.loc[multiplier_p1[defense_columns[0]] == p2_types[0]]
    multiplier_p2 = multiplier_p2.loc[multiplier_p2[defense_columns[0]] == p1_types[0]]
    
    # if 2 types, select second type also, else select NaN
    if p1_types.__len__() < 2:
        multiplier_p2 = multiplier_p2.loc[multiplier_p2[defense_columns[1]].isnull()]
    else:
        multiplier_p2 = multiplier_p2.loc[multiplier_p2[defense_columns[1]] == p1_types[1]]
    if p2_types.__len__() < 2:
        multiplier_p1 = multiplier_p1.loc[multiplier_p1[defense_columns[1]].isnull()]
    else:
        multiplier_p1 = multiplier_p1.loc[multiplier_p1[defense_columns[1]] == p2_types[1]]
        
    multiplier_p1 = multiplier_p1.drop(defense_columns, axis=1)
    multiplier_p2 = multiplier_p2.drop(defense_columns, axis=1)
    
    # take max value, and return
    return [multiplier_p1.values.max(), multiplier_p2.values.max()]

combats['Multiplier_1'] = np.nan
combats['Multiplier_2'] = np.nan

# this dataframe contains ALL columns
combats.loc[:,['Multiplier_1','Multiplier_2']] = list(combats.apply(lambda x: find_multipliers(x['First_pokemon'], x['Second_pokemon']), axis=1))  
  
# safe file
combats.to_csv('../Data/Adjusted data/combined_data.csv',index=False)

# scaling data
combats_scaled = combats.copy()

def scale(column1, column2):
    if 'Legendary' in column1.name:
        x = column1 / column1 + column2
        return x.fillna(.5)
    return column1 / (column1 + column2)

# delete types
del combats_scaled['Type 1_1'], combats_scaled['Type 2_1'], combats_scaled['Type 1_2'], combats_scaled['Type 2_2']
# delete names
del combats_scaled['Name_1'], combats_scaled['Name_2']

# pair columns
paired_columns = {}

for i in combats_scaled.columns:
    if '1' in i:
        paired_columns[i.rsplit('_')[0]] = list(filter(lambda x: i.rsplit('_')[0] in x, combats_scaled.columns))
        
result_df = combats_scaled[['First_pokemon', 'Second_pokemon', 'Winner']].copy()      
for key, value in paired_columns.items():
    result_df[key] = scale(combats_scaled[value[0]], combats_scaled[value[1]])
 
result_df.to_csv('../Data/Adjusted data/combats_scaled_data.csv', index=False)
