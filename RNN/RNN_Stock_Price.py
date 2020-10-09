import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the training set
dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
training_set=dataset_train.iloc[:,1:2].values

#Feature Scaling
# In RNN wherever you have sigmoid function in O/P layer use Normalisation
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)

#Creating a data structure with 60timesteps and 1 output
#Creating x_train and y_train
#x_train will contain 60 previous stock price and y_train will contain stock price of that particular day
x_train=[]
y_train=[]
for i in range(60,1258):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
#x_train contains value form (i-60 to i) if i=60, then values from (0-59)
#y_train gets value at 60.
x_train=np.array(x_train)
y_train=np.array(y_train)

#Reshaping - Adding some more dimensionality
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


#########################################
#Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialising the RNN
regressor=Sequential()
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))

#Adding second LSTM layers with dropout regularization
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

#Adding 3rd LSTM layers
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

#Adding 4th LSTM layer. Suppost thats our last LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#Adding output layer
regressor.add(Dense(units=1))

#Compiling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')

#Fitting RNN to training set
regressor.fit(x_train,y_train,epochs=20,batch_size=32)



#############################################3
#Making the prediction and visualising the result

#Get the real stock price of 2017
dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=dataset_test.iloc[:,1:2].values

#concatenating whole dataset
dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
 
#Creating 3d structure of input
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)
x_test=[]
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
x_test=np.array(x_test)
 
#Getting 3d structure
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predicted_stock_price=regressor.predict(x_test)


#Output we got is scaled, do inverse training for results
predicted_stock_price=sc.inverse_transform(predicted_stock_price)
 

#Visualisng the results
plt.plot(real_stock_price,color='red',label='Real Google stock price')
plt.plot(predicted_stock_price,color='red',label='Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.show()










 
















