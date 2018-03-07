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
    combats = pd.merge(combats,pokemon, right_index=True,left_on='Loser',how='inner',suffixes=('_w','_l')) 
    
    combats=combats.sort_index()
    
    return combats;

#%%
#### function to create plots of the stats in the dataset

def getHistOfStats():
    plt.hist(combats['HP_w'], color = 'green', alpha = 0.75, label = 'winner')
    plt.hist(combats['HP_l'], color = 'red', alpha = 0.75, label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('HP value')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.hist(combats['Attack_w'], color = 'green', alpha = 0.75, label = 'winner')
    plt.hist(combats['Attack_l'], color = 'red', alpha = 0.75, label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Attack value')
    plt.ylabel('Frequency')
    plt.show()

    plt.hist(combats['Defense_w'], color = 'green', alpha = 0.75, label = 'winner')
    plt.hist(combats['Defense_l'], color = 'red', alpha = 0.75, label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Defense value')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.hist(combats['Sp. Atk_w'], color = 'green', alpha = 0.75, label = 'winner')
    plt.hist(combats['Sp. Atk_l'], color = 'red', alpha = 0.75, label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Sp. Atk value')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.hist(combats['Sp. Def_w'], color = 'green', alpha = 0.75, label = 'winner')
    plt.hist(combats['Sp. Def_l'], color = 'red', alpha = 0.75, label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Sp. Def value')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.hist(combats['Speed_w'], color = 'green', alpha = 0.75, label = 'winner')
    plt.hist(combats['Speed_l'], color = 'red', alpha = 0.75, label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Speed value')
    plt.ylabel('Frequency')
    plt.show()
    return;  

#%%
    
def getDensityplotOfStats():
    combats['HP_w'].plot(kind='density',color = 'green',label = 'winner')
    combats['HP_l'].plot(kind='density',color = 'red',label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('HP value')
    plt.ylabel('Frequency')
    plt.show()
    
    combats['Attack_w'].plot(kind='density',color = 'green',label = 'winner')
    combats['Attack_l'].plot(kind='density',color = 'red',label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Attack value')
    plt.ylabel('Frequency')
    plt.show()
    
    combats['Defense_w'].plot(kind='density',color = 'green',label = 'winner')
    combats['Defense_l'].plot(kind='density',color = 'red',label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Defense value')
    plt.ylabel('Frequency')
    plt.show()
    
    combats['Sp. Atk_w'].plot(kind='density',color = 'green',label = 'winner')
    combats['Sp. Atk_l'].plot(kind='density',color = 'red',label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Sp.Atk value')
    plt.ylabel('Frequency')
    plt.show()
    
    combats['Sp. Def_w'].plot(kind='density',color = 'green',label = 'winner')
    combats['Sp. Def_l'].plot(kind='density',color = 'red',label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Defense value')
    plt.ylabel('Frequency')
    plt.show()
    
    combats['Speed_w'].plot(kind='density',color = 'green',label = 'winner')
    combats['Speed_l'].plot(kind='density',color = 'red',label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel('Speed value')
    plt.ylabel('Frequency')
    plt.show()
    
#%%
    
def getCorrelationplotOfStats():
    #Corr plot of pokemon dataset
    stats = pokemon.iloc[:,4:10]

    corr = stats.corr(method='pearson')
    
    print(corr)
    
    plt.matshow(corr,cmap=cm.YlOrRd)
    plt.xticks(range(len(stats.columns)), stats.columns,rotation='vertical')
    plt.yticks(range(len(stats.columns)), stats.columns)
    plt.colorbar()
    plt.suptitle("Correlation plot of all pokemons in pokemon database")
    plt.show()
    
    #Corr plot of combat dataset (Merge both winners and losers first!)
    HP = pd.DataFrame([*combats['HP_w'] , *combats['HP_l']], columns= ['HP'])
    Attack = pd.DataFrame([*combats['Attack_w'] , *combats['Attack_l']], columns= ['Attack'])
    Defense = pd.DataFrame([*combats['Defense_w'] , *combats['Defense_l']],columns= ['Defense'])
    SpAtk = pd.DataFrame([*combats['Sp. Atk_w'] , *combats['Sp. Atk_l']],columns= ['Sp. Atk'])
    SpDef = pd.DataFrame([*combats['Sp. Def_w'] , *combats['Sp. Def_l']],columns= ['Sp. Def'])
    Speed = pd.DataFrame([*combats['Speed_w'] , *combats['Speed_l']],columns= ['Speed'])
   
    cStats=HP.join(Attack.join(Defense.join(SpAtk.join(SpDef.join(Speed)))))
    
    corr2 = cStats.corr(method='pearson')
    
    print(corr2)
    
    plt.matshow(corr2,cmap=cm.YlOrRd)
    plt.xticks(range(len(stats.columns)), stats.columns,rotation='vertical')
    plt.yticks(range(len(stats.columns)), stats.columns)
    plt.colorbar()
    plt.suptitle("Correlation plot of all pokemons in combat database")
    plt.show()
    
#%%
  
def legendaryAnalysis():
    #Not finished yet!
    print(combats.groupby(['Legendary_w'])['Winner'].count())
    print(combats.groupby(['Legendary_l'])['Loser'].count())
    
    objects = ('Legendary','Non legendary' )
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos,combats.groupby(['Legendary_w'])['Winner'].count())
    plt.xticks(y_pos, objects)
    plt.ylabel('Amount')
    plt.title('Wins Legendary')

#%%    

###### Paste the stats of the pokemons in the combats file

#To run from this place, press F9
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

#Load data
combats = pd.read_csv('./Documents/Github/MachineLearning/Data/Original data/combats.csv',delimiter =',')
pokemon = pd.read_csv('./Documents/Github/MachineLearning/Data/Original data/pokemon.csv',delimiter =',')
    
#Process (combat) data 
combats = getProcessedCombatData(combats,pokemon)

#Get densityplots of stats
getDensityplotOfStats()

#Get histograms
getHistOfStats()                        
                            
#Get correlationplot of all 6 stats
getCorrelationplotOfStats()

#next function



