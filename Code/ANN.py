# -*- coding: utf-8 -*-
"""
Spyder Editor
@author: joy
"""

#Import packages
import numpy as np
import pandas as pd

#Load dataset
data=pd.read_csv("file:///E:/Data Sets/Loan Prediction/train_u6lujuX_CVtuZ9i.csv")

#List of column names
list(data)

#Types of data columns
data.dtypes

#Sample of data
data.head(10)

#(2)DATA CLEANING AND PREPROCESSING
#Find missing values
data.isnull().sum()

#Impute missing values with mean (numerical variables)
data.fillna(train.mean(),inplace=True) 
data.isnull().sum() 

#Impute missing values with mode (categorical variables)
data.Gender.fillna(data.Gender.mode()[0],inplace=True)
data.Married.fillna(data.Married.mode()[0],inplace=True)
data.Dependents.fillna(data.Dependents.mode()[0],inplace=True) 
data.Self_Employed.fillna(data.Self_Employed.mode()[0],inplace=True)  
data.isnull().sum()  

#(3)BUILDING ARTIFICIAL NEURAL NETWORK
#Remove Loan_ID variable - Irrelevant
data=data.drop('Loan_ID',axis=1)

#Create target variable
X=data.drop('Loan_Status',1)
y=data.Loan_Status

#Build dummy variables for categorical variables
X=pd.get_dummies(X)
data=pd.get_dummies(data)

#Split train data for cross validation
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#Train model
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=(15,10,5))
mlp.fit(x_train,y_train)

#Make predictions
pred=mlp.predict(x_test)

#Evaluate accuracy of model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
accuracy_score(y_test,pred) #77.24%
confusion_matrix(y_test,pred)
