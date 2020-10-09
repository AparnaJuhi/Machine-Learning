import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('Different share market.csv')
X=dataset.iloc[:,1:9].values
Y=dataset.iloc[:,9].values
#Splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)


#applying linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the test_Set Results
Y_pred=regressor.predict(X_test)
