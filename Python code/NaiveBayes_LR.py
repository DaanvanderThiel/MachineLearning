# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 12:52:42 2018
Script voor het splitsen van de train en test set
@author: daan
"""
#%%
"""Imports only"""
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import naive_bayes, linear_model
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import interp
import itertools
#%%
import os
os.chdir('Documents/GitHub/MachineLearning/Python code')

#%%
"""Splitting the dataset"""
data = pd.read_csv('../Data/Adjusted data/combats_scaled_data.csv')

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:12], data["WinnerBinary"], train_size=0.7, random_state=42)
#%%
# correlation matrix (does not prove independece)
df_copy = X_train.copy()
df_copy = df_copy.iloc[:, 3:12]
corr2 = df_copy.corr(method='pearson')
print(corr2)

plt.matshow(corr2,cmap=cm.Blues)
plt.xticks(range(len(df_copy.columns)), df_copy.columns,rotation='vertical')
plt.yticks(range(len(df_copy.columns)), df_copy.columns)
plt.colorbar()
plt.suptitle("Correlation plot of all pokemons in combat database")
plt.show()
#%%
# function to train a model naive bayes
def train_NB(x_train, y_train):
    model = naive_bayes.GaussianNB()
    model.fit(X=x_train, y=y_train)
    
    return model

# function to train model logistic regression
def train_LR(x_train, y_train, kwargs):
    model = linear_model.LogisticRegression(**kwargs)
    model.fit(X=x_train, y=y_train)
    
    return model

def grid_search(C_range, x_train, y_train, x_validate, y_validate):
    models = dict()
    penalties = ['l1','l2']
    for i in  C_range:
        for j in penalties:
            tmp = train_LR(x_train, y_train, dict(C=i, penalty=j))
            models['C_{}_penalty_{}'.format(i, j)] = tmp.score(x_validate, y_validate)
    return models              
    
def plot_ROC(method, models, x_validate, y_validate):
    plt.title('Receiver Operating Characteristic')
    for i in models:
        x = model.predict_proba(X=x_validate)
        fpr, tpr, thresholds = roc_curve(y_validate, x[:,1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='{}, AUC = %0.2f'.format(method)%roc_auc)        
    pass

def cross_validation(model, X, y, folds, kwargs):
    model = model(**kwargs)
    cv = StratifiedKFold(n_splits=folds)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(10,8))
    plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.plot([0,1],[0,1],'k--')
    i = 0
    for train, test in cv.split(X, y):   
        probas_ = model.fit(X=X.iloc[train], y=y.iloc[train]).predict_proba(X=X.iloc[test])
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
    plt.legend()
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.xlabel('False Positive Rate', fontsize=12)
    return mean_fpr, mean_tpr, std_auc
    
def grid_search_NB(X, y, n_features):
    """Search among features"""
    model = naive_bayes.GaussianNB()
    plt.figure(figsize=(10,8))
    plt.title('Receiver Operating Characteristic curve: {} feature(s)'.format(n_features), fontsize=16)
    X_train, X_validate, y_train, y_validate = train_test_split(X.iloc[:,:12], y, train_size=4/7, random_state=42)
    for j in itertools.combinations(X.columns.values[3:],r=n_features):
        # cross_validation(model, X[list(j)], y, folds = 5, kwargs=dict(name=j))
        model.fit(X=X_train[list(j)], y=y_train)
        x = model.predict_proba(X=X_test[list(j)])
        fpr, tpr, thresholds = roc_curve(y_test, x[:,1])
        roc_auc = auc(fpr, tpr)
        if n_features == 1:
            plt.plot(fpr, tpr, label='{}, AUC = %0.2f'.format(list(j)[0])%roc_auc)  
        elif n_features == 2:
            #plt.plot(fpr, tpr, label='{}, {}, AUC = %0.2f'.format(list(j)[0], list(j)[1])%roc_auc)  
            if roc_auc > 0.8:
                plt.plot(fpr, tpr, label='{}, {} AUC = %0.2f'.format(list(j)[0], list(j)[1])%roc_auc)  
        elif n_features == 3:
            if roc_auc > 0.8:
                plt.plot(fpr, tpr, label='{}, {}, {}, AUC = %0.2f'.format(list(j)[0], list(j)[1], list(j)[2])%roc_auc)  
        #if roc_auc > 0.8:
        #    plt.plot(fpr, tpr, label='{}, {}, {}, AUC = %0.2f'.format(list(j)[0], list(j)[1], list(j)[2])%roc_auc)  
    plt.legend()
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.xlabel('False Positive Rate', fontsize=12)
     
    
#%%
# Trying Naive bayes with 1 - 3 features
grid_search_NB(X_train, y_train, 1)
grid_search_NB(X_train, y_train, 2)
grid_search_NB(X_train, y_train, 3)
#%%
# No better model than model with 1 feature: 'Speed' so cross validation to ensure stability of model
mean_fpr, mean_tpr, sd_auc = cross_validation(naive_bayes.GaussianNB, pd.DataFrame(X_train['Speed']), 
                                              y_train,folds=10, kwargs={})
#%%    
cross_validation(linear_model.LogisticRegression, X_train, y_train, 10, dict(C=0.01, penalty='l1'))
#%%
# do grid search on logistic regression, all data
x = grid_search(C_range=np.arange(0.01,2,0.01), x_train=X_train, y_train=y_train, 
                x_validate=X_validate, y_validate=y_validate)
#%%
# best model (based on accuracy)
print(max(x.values()))
# 0,8983 so about 89,8% this is model with value C=0,01 and penalty 'l1'

#%%
# now try same, but with only 2 parameters (Attack and Speed)
x2 = grid_search(C_range=np.arange(0.01,2,0.01), x_train=X_train.loc[:, ['Attack', 'Speed']], y_train=y_train, 
                x_validate=X_validate.loc[:, ['Attack', 'Speed']], y_validate=y_validate)
#%%
# best model (based on accuracy)
print(max(x2.values()))
# 0,9071 so about 90,7% this is model with C=0,01 and penalty 'l1'
#%%
model = train_LR(x_train=X_train.loc[:, ['Attack', 'Speed']], y_train=y_train, kwargs=dict(C=0.01, penalty='l1'))
predictions = model.predict_proba(X=X_validate.loc[:, ['Attack', 'Speed']])
actual = y_validate
false_positive_rate_LR, true_positive_rate_LR, thresholds_LR = roc_curve(actual, predictions[:,1])
roc_auc_LR = auc(false_positive_rate_LR, true_positive_rate_LR)

#%%
# Logistic Regression
LR = linear_model.LogisticRegression()
LR.fit(X_train.loc[:,X_train.columns[3:]], y=y_train)
#LR.score(X=X_validate.loc[:,X_validate.columns[3:]], y=y_validate)
#%%
# from correlation matrix see that Sp. Atk and Sp. Def have some correlation, so leave these features out
without_special = list(filter(lambda x: x != 'Sp. Atk' and x != 'Sp. Def',X_train.columns))
LR.fit(X=X_train.loc[:, without_special], y=y_train)
LR.score(X=X_validate.loc[:,without_special], y=y_validate)
# does not significantly worsen the accuracy of the model.

