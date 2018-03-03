# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


    
#%%   
    
def getProcessedCombatData(combats,pokemon):  

    pokemon = pokemon.set_index('#')
                                
    #Add a column with the ID of the loser
    combats['Loser'] = [combats['First_pokemon'].iloc[i] 
        if combats['First_pokemon'].iloc[i] != 
            combats['Winner'].iloc[i] else combats['Second_pokemon'].iloc[i] 
            for i in combats.index]
    
    #Add the stats of the winner behind his ID in seperate columns
    combats = pd.merge(combats,pokemon, right_index=True,left_on='Winner',how='inner') 
    
    #Add the stats of the loser behind his ID in seperate columns
    combats = pd.merge(combats,pokemon, right_index=True,left_on='Loser',how='inner') 
    
    return combats;

#%%
#### function to create plots of the stats in the dataset

def getHistOfStats():
    plt.hist(combats['HP_x'], color = 'green', alpha = 0.75, label = 'winner')
    plt.hist(combats['HP_y'], color = 'red', alpha = 0.75, label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('HP value')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.hist(combats['Attack_x'], color = 'green', alpha = 0.75, label = 'winner')
    plt.hist(combats['Attack_y'], color = 'red', alpha = 0.75, label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Attack value')
    plt.ylabel('Frequency')
    plt.show()

    plt.hist(combats['Defense_x'], color = 'green', alpha = 0.75, label = 'winner')
    plt.hist(combats['Defense_y'], color = 'red', alpha = 0.75, label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Defense value')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.hist(combats['Sp. Atk_x'], color = 'green', alpha = 0.75, label = 'winner')
    plt.hist(combats['Sp. Atk_y'], color = 'red', alpha = 0.75, label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Sp. Atk value')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.hist(combats['Sp. Def_x'], color = 'green', alpha = 0.75, label = 'winner')
    plt.hist(combats['Sp. Def_y'], color = 'red', alpha = 0.75, label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Sp. Def value')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.hist(combats['Speed_x'], color = 'green', alpha = 0.75, label = 'winner')
    plt.hist(combats['Speed_y'], color = 'red', alpha = 0.75, label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Speed value')
    plt.ylabel('Frequency')
    plt.show()
    return;  

#%%
    
def getDensityplotOfStats():
    combats['HP_x'].plot(kind='density',color = 'green',label = 'winner')
    combats['HP_y'].plot(kind='density',color = 'red',label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('HP value')
    plt.ylabel('Frequency')
    plt.show()
    
    combats['Attack_x'].plot(kind='density',color = 'green',label = 'winner')
    combats['Attack_y'].plot(kind='density',color = 'red',label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Attack value')
    plt.ylabel('Frequency')
    plt.show()
    
    combats['Defense_x'].plot(kind='density',color = 'green',label = 'winner')
    combats['Defense_y'].plot(kind='density',color = 'red',label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Defense value')
    plt.ylabel('Frequency')
    plt.show()
    
    combats['Sp. Atk_x'].plot(kind='density',color = 'green',label = 'winner')
    combats['Sp. Atk_y'].plot(kind='density',color = 'red',label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Sp.Atk value')
    plt.ylabel('Frequency')
    plt.show()
    
    combats['Sp. Def_x'].plot(kind='density',color = 'green',label = 'winner')
    combats['Sp. Def_y'].plot(kind='density',color = 'red',label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Defense value')
    plt.ylabel('Frequency')
    plt.show()
    
    combats['Speed_x'].plot(kind='density',color = 'green',label = 'winner')
    combats['Speed_y'].plot(kind='density',color = 'red',label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Speed value')
    plt.ylabel('Frequency')
    plt.show()
    
#%%
    
def getCorrelationplotOfStats():
    stats = pokemon.iloc[:,4:10]

    corr = stats.corr(method='pearson')
    
    plt.matshow(corr,cmap=cm.YlOrRd)
    plt.xticks(range(len(stats.columns)), stats.columns,rotation='vertical')
    plt.yticks(range(len(stats.columns)), stats.columns)
    plt.colorbar()
    plt.show()
    
    x1 = pd.DataFrame([*combats['HP_x'] , *combats['HP_y']], columns= ['x1'])
    x2 = pd.DataFrame([*combats['Attack_x'] , *combats['Attack_y']], columns= ['x2'])
    x3 = pd.DataFrame([*combats['Defense_x'] , *combats['Defense_y']],columns= ['x3'])
    x4 = pd.DataFrame([*combats['Sp. Atk_x'] , *combats['Sp. Atk_y']],columns= ['x4'])
    x5 = pd.DataFrame([*combats['Sp. Def_x'] , *combats['Sp. Def_y']],columns= ['x5'])
    x6 = pd.DataFrame([*combats['Speed_x'] , *combats['Speed_y']],columns= ['x6'])
   
    z=pd.merge(x1,x2)
    
    
    corr2 = y.corr(method='pearson')
    
#%%

###### Paste the stats of the pokemons in the combats file

#To run from this place, press F9
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

#Load data
combats = pd.read_csv('C:/Users/Jesse/Documents/Github/MachineLearning/Data/Original data/combats.csv',delimiter =',')
pokemon = pd.read_csv('C:/Users/Jesse/Documents/Github/MachineLearning/Data/Original data/pokemon.csv',delimiter =',')
    
#Process (combat) data 
combats = getProcessedCombatData(combats,pokemon)

#Get densityplots of stats
getDensityplotOfStats()

#Get histograms
getHistOfStats()                        
                            
#Get correlationplot of all 6 stats
getCorrelationplotOfStats()





