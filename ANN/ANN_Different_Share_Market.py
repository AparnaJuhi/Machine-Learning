import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Data Preprocessing
dataset=pd.read_csv('Different share market.csv')
X=dataset.iloc[:,1:9].values
Y=dataset.iloc[:,9].values

#Splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

#Lets make an ANN by importing keras
from keras.models import Sequential
from keras.layers import Dense

# Sequential is used to initialise NN
#Dense is used to make lauers of NN
regressor=Sequential()

#Adding the input layer and the first hidden layer
#Step 1: Randomly initialise the  weights to number close to 0.
#It will be taken care by Dense
# We can only predict the no of nodes in hiddne layer. Avg of no of nodes  in input and output layer (11+1)/2=6.
#uniform func to initialise weights small no close to 0.


#Step 2: Input first observation of your dataset in the input layer, each feature in one input node
# No. of nodes in input layer will be 8, as we have 8 attributes in ANN- output_dim=(8+1)/2

#Step 3: Sigmoid is used for output lauer and Relu activation function is used for hidden layer


regressor.add(Dense(output_dim=5,init='uniform',activation='relu',input_dim=8))

#Adding output layer, output_dim=1 as we want only one output
regressor.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#compiling the ANN, adam is an algo of stochastic gradient descent
regressor.compile(optimizer='adam', loss='mean_squared_error')


#Step 4: Compare the predicted result with output result. Measure the generated error
#Step 5: Back Propagation from left to right
#Step 6: Update weights after each obs
regressor.fit(X_train,Y_train,batch_size=10,nb_epoch=50)

#Making prediction 
Y_pred=regressor.predict(X_test)
