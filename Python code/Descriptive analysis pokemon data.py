# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%

#### function to create plots of the stats in the dataset

def histOfStats(file,statNr,statName):
    winner = [file['winner_stats'].iloc[i][statNr] for i in range(len(file))]
    loser = [file['loser_stats'].iloc[i][statNr] for i in range(len(file))]

    plt.hist(x=winner,color='green',alpha=0.75,label='winner')
    plt.hist(x=loser,color='red',alpha=0.75,label='loser')
    plt.legend(loc = 'upper right')
    statString = statName
    plt.xlabel(statString)
    plt.ylabel('Frequency')
    plt.show()
    
    return;
#%%

###### Paste the stats of the pokemons in the combats file

#To run from this place, press F9
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

combats = pd.read_csv('C:/Users/Jesse/Documents/Github/MachineLearning/Data/Original data/combats.csv',delimiter =',')
pokemon = pd.read_csv('C:/Users/Jesse/Documents/Github/MachineLearning/Data/Original data/pokemon.csv',delimiter =',')

#Set the index of the pokemon table equal to the ID of the pokemon
pokemon = pokemon.set_index('#')

#Add the stats of the winner behind his ID
combats['winner_stats'] = [pokemon.loc[i] for i in combats['Winner']]

#Add a column with the ID of the loser
combats['Loser'] = [combats['First_pokemon'].iloc[i] 
    if combats['First_pokemon'].iloc[i] != 
        combats['Winner'].iloc[i] else combats['Second_pokemon'].iloc[i] 
        for i in combats.index]

#Add the stats of the loser behind his ID
combats['loser_stats'] = [pokemon.loc[i] for i in combats['Loser']]

#%%

#### Old hist code!

attack_winner = [combats['winner_stats'].iloc[i].Attack for i in range(len(combats))]
attack_loser = [combats['loser_stats'].iloc[i].Attack for i in range(len(combats))]

plt.hist(x=attack_winner, color = 'green', alpha = 0.75, label = 'winner')
plt.hist(x=attack_loser, color = 'red', alpha = 0.75, label = 'loser')
plt.legend(loc = 'upper right')
plt.xlabel('Attack value')
plt.ylabel('Frequency')
plt.show()

#Comparing defense winner/loser
defense_winner = [combats['winner_stats'].iloc[i].Defense for i in range(len(combats))]
defense_loser = [combats['loser_stats'].iloc[i].Defense for i in range(len(combats))]

plt.hist(x=defense_winner, color = 'green', alpha = 0.75, label = 'winner')
plt.hist(x=defense_loser, color = 'red', alpha = 0.75, label = 'loser')
plt.legend(loc = 'upper right')
plt.xlabel('Defense value')
plt.ylabel('Frequency')
plt.show()

meanSpeed = np.mean([combats['winner_stats'].iloc[i].Speed for i in range(len(combats))])

#%%

#Generate histograms:
statdictionary = {3:'HP', 4:'Attack',5: 'Defense',6: 'Sp. Atk', 7:'Sp. Def',8:'Speed', 9:'Generation',10:'Legendary'}
for x in range(3,10):
    histOfStats(combats,x,statdictionary[x])

