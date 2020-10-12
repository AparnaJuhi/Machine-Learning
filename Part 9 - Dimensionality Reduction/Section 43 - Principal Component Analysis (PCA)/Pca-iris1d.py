import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv("iris.data",names=['sepal length','sepal width','petal length','petal width','target'])
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values



from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)

from sklearn.decomposition import PCA
pca=PCA(n_components=1)
pc_X=pca.fit_transform(X)

PCA_dataset=pd.DataFrame(data=pc_X,columns=['Principal component1'])
final_dataset=pd.concat([PCA_dataset,dataset[['target']]],axis=1)
fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1')
ax.set_title('1 component PCA')
ll=len(Y)
A=[]
B=[]
C=[]
for k in range(ll):
    if(Y[k] == 'Iris-setosa'):
        A.append(k)
    if(Y[k] =='Iris-versicolor'):
        B.append(k)
    if(Y[k] =='Iris-virginica'):
        C.append(k)
    
plt.plot(A, '*',color='r')
plt.plot(B, '*',color='b')
plt.plot(C, '*',color='g')
plt.show()



from sklearn.preprocessing import LabelEncoder
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(pc_X,Y,test_size=0.4,random_state=0)


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier()
classifier.fit(X_train,Y_train)

#Predicting the test_Set Results
Y_pred=classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)


from sklearn import metrics
metrics.accuracy_score(Y_test,Y_pred)


