# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 12:52:42 2018
Script voor het splitsen van de train en test set
@author: daan
"""

 from sklearn.model_selection import train_test_split
 import numpy as np # linear algebra
 import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
 X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:12], data["WinnerBinary"], train_size=0.7, random_state=42)
 
 # split test set in test en validate set met test is 40% en validatie is 30%
 
 X_train, X_validate, y_train, y_validate = train_test_split(X_train.iloc[:,:12], y_train, train_size=4/7, random_state=42)