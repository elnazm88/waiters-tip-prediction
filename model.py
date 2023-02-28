import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import pickle
#reading dataset
data=pd.read_csv("tips.csv")
#data.isnull().sum() #indicate number of null values
#convert categorical features to numerical
data['sex'] = data['sex'].map({'Male':0, 'Female':1})
data['day'] = data['day'].map({'Thur':0, 'Fri':1, 'Sat':2, 'Sun':3})
data['time'] = data['time'].map({'Lunch':0, 'Dinner':1})
data['smoker'] = data['smoker'].map({'No':0, 'Yes':1})
# Drop the smoker feature as it has a low correlation 
data.drop(['smoker'], axis=1, inplace=True)
# Data Labeling
class_labels = data['tip'].values 
data.drop(['tip'], axis=1, inplace=True) 
#splitting dataset to train and test dataset
X = data.values
y= class_labels
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
svreg = SVR(kernel='rbf').fit(X_train, y_train)
# save the model
pickle.dump(svreg, open('tips1.pkl', 'wb'))