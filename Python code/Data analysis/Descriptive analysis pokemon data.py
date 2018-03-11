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
    
def getDensityplotOfStats(columnName1,columnName2,columnName3):
    combats[columnName1].plot(kind='density',color = 'green',label = 'winner')
    combats[columnName2].plot(kind='density',color = 'red',label = 'loser')
    plt.legend(loc = 'upper right')
    plt.xlabel(columnName3 + ' value', fontweight="bold")
    plt.ylabel('Density', fontweight="bold")
    plt.xlim([0,max(combats[columnName1].max(), combats[columnName2].max())*1.25])
    plt.title('Density plot ' + columnName3 + ' (winner vs loser)', fontweight="bold")
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
    
def crossTableWL(columnName1,columnName2,columnName3):
    #outputs a dataframe
    
    Win = combats.groupby([columnName1]).size().rename("Win")
    Loss = combats.groupby([columnName2]).size().rename("Loss")
    
    temp = Win.to_frame().join(Loss)
    
    x1 = round(Win / (Win + Loss)*100,2)
    x2 = round(Loss / (Win + Loss)*100,2)
    Perc=pd.concat([x1,x2],axis=1)
    
    result = pd.merge(temp,Perc, left_index = True, right_index = True)
    result.columns = ['Win', 'Loss', 'Win %', 'Loss %']
    
    result.index.name = columnName3

    print(result)
    
    return result;
#%%
''' 
def legendaryAnalysis():
    #Not finished yet!
    print(combats.groupby(['Legendary_w'])['Winner'].count())
    print(combats.groupby(['Legendary_l'])['Loser'].count())
    
    objects = ('Non legendary','Legendary' )
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos,combats.groupby(['Legendary_w'])['Winner'].count()/len(combats)*100, color = ['red', 'blue'])
    plt.xticks(y_pos, objects)
    plt.ylabel('Percentage (%)')
    plt.title('Proportion Legendary/Non-Legendary in winners')
    plt.show()

    #plt.bar(y_pos,pokemon.query("Legendary == True")[['Legendary']].count(),  pokemon.query("Legendary == False")[['Legendary']].count(), color = ['red', 'blue'])
    
    plt.bar(y_pos,pokemon.groupby(['Legendary'])['Legendary'].count()/len(pokemon)*100, color = ['red', 'blue'])
    plt.xticks(y_pos, objects)
    plt.ylabel('Percentage (%)')
    plt.title('Proportion Legendary/Non-Legendary in dataset')
    plt.show()
    
    crossTableWL('Legendary_w','Legendary_l')
'''
#%%    

###### Paste the stats of the pokemons in the combats file

#To run from this place, press F9
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

#Load data
combats = pd.read_csv('C:/Users/Elitebook/Desktop/Machine Learning/Data/Original data/combats.csv',delimiter =',')
pokemon = pd.read_csv('C:/Users/Elitebook/Desktop/Machine Learning/Data/Original data/pokemon.csv',delimiter =',')
    
#Process (combat) data 
combats = getProcessedCombatData(combats,pokemon)

#Get densityplots of stats
getDensityplotOfStats('HP_w', 'HP_l', 'HP')
getDensityplotOfStats('Attack_w', 'Attack_l', 'Attack')
getDensityplotOfStats('Defense_w', 'Defense_l', 'Defense')
getDensityplotOfStats('Sp. Atk_w', 'Sp. Atk_l', 'Sp.Atk')
getDensityplotOfStats('Sp. Def_w', 'Sp. Def_l', 'Sp.Def')
getDensityplotOfStats('Speed_w', 'Speed_l', 'Speed')

#Get correlationplot of all 6 stats
getCorrelationplotOfStats()

#Crosstables of Type, Legendary and Generation
Legendary = crossTableWL('Legendary_w','Legendary_l', 'Legendary')
Type= crossTableWL('Type 1_w','Type 1_l', 'Type')
Generation = crossTableWL('Generation_w','Generation_l', 'Generation')

firstWin = combats[(combats.First_pokemon == combats.Winner)]['First_pokemon'].count()
secondWin = combats[(combats.First_pokemon != combats.Winner)]['First_pokemon'].count()
firstWinPercentage = round(firstWin / (firstWin + secondWin) * 100, 2)
secondWinPercentage = round(secondWin / (firstWin + secondWin) * 100, 2) 



#legendaryAnalysis()
#Get histograms
#getHistOfStats()