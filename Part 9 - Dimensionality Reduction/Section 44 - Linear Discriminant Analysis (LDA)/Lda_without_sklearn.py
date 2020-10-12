import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv("iris.data",names=['sepal length','sepal width','petal length','petal width','target'])
dataset.dropna(how='all',inplace=True)
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values


from sklearn.preprocessing import LabelEncoder
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#Finding mean of each column+Each class
mean_vectors_list=[]
for i in range(0,3):
    for j in range(0,len(X)):
        mean_vec=np.mean(X[Y==i], axis=0)
    mean_vectors_list.append(mean_vec)


#Within class scatter matrix
A=[]
for i in range(0,len(X)):
    mat_A=[]
    for j in range(0,4):
        mat_A.append(0)
    A.append(mat_A)
    


for k in range(0,3):
    for i in range(0,len(X)):
        for j in range(0,4):
            if(Y[i]==k):
                A[i][j]=X[i][j]-mean_vectors_list[k][j]
                
B=np.array([np.array(xi) for xi in A])
#Alternatively, we could also compute the class-covariance matrices by adding the scaling factor N−1 to the within-class scatter matrix, so that our equation becomes
N=50
cov_mat=(B).T.dot(B)/(N-1)




#Between class scatter matrix
overall_mean = np.mean(X, axis=0)

S_B = np.zeros((4,4))
for i,mean_vec in enumerate(mean_vectors_list):  
    n = X[Y==i+1,:].shape[0]
    mean_vec = mean_vec.reshape(4,1) # make column vector
    overall_mean = overall_mean.reshape(4,1) # make column vector
    S_B += 50 * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

eigen_value, eigen_vector = np.linalg.eig(np.linalg.inv(cov_mat).dot(S_B))
print(eigen_value)
print(eigen_vector)


eig_pairs=[]
for i in range(len(eigen_value)):
    eig_pairs.append((np.abs(eigen_value[i]),eigen_vector[:,i]))
print(eig_pairs)

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()
tot = sum(eigen_value)
var_exp = [(i / tot)*100 for i in sorted(eigen_value, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
a=[1,2,3,4]
aa= list(range(len(a)))
# plot the index for the x-values
plt.plot(aa, cum_var_exp, marker='o', linestyle='--', color='r', label='Circle') 
plt.xticks(aa,a)
plt.xlabel('Principal Components')
plt.ylabel('Explained variance in %') 
plt.title('Visualising the results for different PCA')
plt.legend() 
plt.show()
"""
print('between-class Scatter Matrix:\n', S_B)
#make a 3*4 matrix of elements 0
A=[]
for i in range(0,4):
    mat_A=[]
    for j in range(0,3):
        mat_A.append(0.0)
    A.append(mat_A)
    
A=np.array([np.array(xi) for xi in A])    
mean_vectors_list=np.array([np.array(xi) for xi in mean_vectors_list])
mean_vectors_list=mean_vectors_list.reshape(4,3)
overall_mean = np.mean(X, axis=0)
for i in range(0,4):
    for j in range(0,3):
        A[i][j]=mean_vectors_list[i][j]-overall_mean[i]
between_class=(A).dot((A).T)      


eigen_value,eigen_vector=np.linalg.eig(cov_mat)
print(eigen_value)
print(eigen_vector)
"""

matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[1][1].reshape(4,1)))

Projection=X.dot(matrix_w)

PCA_dataset=pd.DataFrame(data=Projection,columns=['Principal component1','Principal Component 2'])
final_dataset=pd.concat([PCA_dataset,dataset[['target']]],axis=1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('2 component PCA')
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['#0D76BF', '#00cc96', '#EF553B']
for target, color in zip(targets,colors):
    indicesToKeep = final_dataset['target'] == target
    ax.scatter( final_dataset.loc[indicesToKeep, 'Principal component1']
               ,  final_dataset.loc[indicesToKeep, 'Principal Component 2']
               , c = color
               , s = 50)
    
ax.legend(targets)
ax.grid()
