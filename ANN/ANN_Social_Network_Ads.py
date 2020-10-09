import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Data Preprocessing
dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,1:4].values
Y=dataset.iloc[:,4].values


#Encode these categorical data i.e Gender in this case
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#Encoding the Geography attribute
labelencoder=LabelEncoder()
X[:,0]=labelencoder.fit_transform(X[:,0])

#Splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#implement feature scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


#Lets make an ANN by importing keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# Sequential is used to initialise NN
#Dense is used to make lauers of NN
classifier=Sequential()

#Adding the input layer and the first hidden layer
#Step 1: Randomly initialise the  weights to number close to 0.
#It will be taken care by Dense
# We can only predict the no of nodes in hiddne layer. Avg of no of nodes  in input and output layer (11+1)/2=6.
#uniform func to initialise weights small no close to 0.


#Step 2: Input first observation of your dataset in the input layer, each feature in one input node
# No. of nodes in input layer will be 11, as we have 11 attributes in ANN

#Step 3: Sigmoid is used for output lauer and Relu activation function is used for hidden layer


classifier.add(Dense(output_dim=2,init='uniform',activation='relu',input_dim=3))

#Adding output layer, output_dim=1 as we want only one output
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#compiling the ANN, adam is an algo of stochastic gradient descent
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#Step 4: Compare the predicted result with output result. Measure the generated error
#Step 5: Back Propagation from left to right
#Step 6: Update weights after each obs
classifier.fit(X_train,Y_train,batch_size=10,nb_epoch=50)

#Making prediction 
Y_pred=classifier.predict(X_test)

#the answer should be in true/false not in the form of fraction
Y_pred=Y_pred>0.5
#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)
print("Confusion matrix : ")
print(cm)

from sklearn.metrics import accuracy_score
print("Accuracy : ")
print(accuracy_score(Y_test,Y_pred))