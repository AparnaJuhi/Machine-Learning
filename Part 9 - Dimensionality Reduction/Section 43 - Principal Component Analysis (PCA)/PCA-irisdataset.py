import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv("iris.data",names=['sepal length','sepal width','petal length','petal width','target'])
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)


from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pc_X=pca.fit_transform(X)


PCA_dataset=pd.DataFrame(data=pc_X,columns=['Principal component1','Principal Component 2'])
final_dataset=pd.concat([PCA_dataset,dataset[['target']]],axis=1)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('2 component PCA')
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = final_dataset['target'] == target
    ax.scatter( final_dataset.loc[indicesToKeep, 'Principal component1']
               ,  final_dataset.loc[indicesToKeep, 'Principal Component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()




from sklearn.preprocessing import LabelEncoder
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

