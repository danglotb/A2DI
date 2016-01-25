
import scipy.cluster.vq as sci
import sklearn.datasets as sk
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2

moon=sk.make_circles(shuffle=False,  noise=0.025)

P=4
Epsilon = 0.1
K=2

def label_to_color(label):
	if label == "Iris-setosa":
		return 'red'
	elif label == "Iris-versicolor":
		return 'green'
	elif label == "Iris-virginica":
		return 'blue'
	else:
		return 'black'

def s(i, j, sigma):
	dist = sum([ pow(t[0]-t[1], 2) for t in zip(i,j)])
	return math.exp(-dist/(sigma*sigma))

def mykmeans(X,Y,ORI):
	ret,index=kmeans2(X,K)
	color_tab=['r', 'b', 'g']
	plt.scatter(ORI.T[0],ORI.T[1], color=[color_tab[i] for i in index])
	plt.show()
	err=np.zeros((K,K))
	for i in range(len(index)):
		for k1 in range(K):
			if k1 == Y[i]:
				err[index[i],k1] += 1
	map_index=[0]*K
	for k in range(K):
		map_index[k] = np.argmax(err[k])
	err=[0]*K
	for i in range(len(Y)):	
		if not Y[i] == map_index[index[i]]:
			err[Y[i]] += 1
	return err


#moon=sk.make_moons(shuffle=True, noise=0.2)

data = [line.rstrip('\n').split(',') for line in open('iris.data')]
X = [[float(y) for y in d[:P]] for d in data]
Label = [0 if d[P] == "Iris-setosa" else 1 if d[P] == "Iris-versicolor" else 2 for d in data]
Label_color = [label_to_color(l) for l in Label]

X = moon[0]
Label = moon[1]
sigma=0.09

#Matrice d'adjacence avec sparsification
S = np.zeros((len(X), len(X)))
D = np.zeros((len(X), len(X)))
for i in range(len(X)):
	count = 0
	for j in range(len(X)):
		if not i == j:
			val = s(X[i], X[j], sigma)
			if val > 0:#Epsilon:
				S[i,j] = val
				count += 1#val
	D[i,i] = count

Laplacian = np.subtract(D,S)
w,u=np.linalg.eig(Laplacian)
index_order=np.argsort(w)
u_sorted=u[index_order]
w_sorted=w[index_order]

err_kmeans=[0]*K
err_spectre=[0]*K

nbrun=1


for _ in range(nbrun):
	err_kmeans=np.add(err_kmeans,mykmeans(np.matrix(X), Label, np.matrix(X)))
	err_spectre=np.add(err_spectre,mykmeans(u[:,1:K], Label, np.matrix(X)))

print(np.divide(err_kmeans,nbrun))
print(np.divide(err_spectre,nbrun))
