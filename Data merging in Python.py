# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#To run from this place, press F9
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Histogram of the winners
data.plot(kind = "hist",y = "winning_Number")

data["winning_Number"] = data["winning_Number"].astype("category")

#Set the index of the pokemon table equal to the ID of the pokemon
pokemon = pokemon.set_index('ID')

#Add the stats of the winner behind his ID
combats['winner_stats'] = [pokemon.loc[i] for i in combats['Winner']]

#Add a column with the ID of the loser
combats['Loser'] = [combats['First_pokemon'].iloc[i] 
    if combats['First_pokemon'].iloc[i] != 
        combats['Winner'].iloc[i] else combats['Second_pokemon'].iloc[i] 
        for i in combats.index]

#Add the stats of the loser behind his ID
combats['loser_stats'] = [pokemon.loc[i] for i in combats['Loser']]

meanSpeed = np.mean([combats['winner_stats'].iloc[i].Speed for i in range(len(combats))])
