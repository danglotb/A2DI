
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import random as r

#fitness
def j(N, y, theta, x):
	return sum( pow(y - (theta[0,0]*x + theta[1,1]), 2)) / N
	
#fonction f to compute y from x
def f(x, tone, ttwo, e):
	return tone + ttwo * x + e

#produce data set, learn it and test the learning by retourning the j fun 
def compute(N, b, theta1, theta2, mu, sigma):

	x = []
	y = []

	while len(x) != N:
		rand = r.random()*b - b/2
		x.append(rand)
		y.append(f(rand, theta1, theta2, np.random.normal(mu, sigma)))

	ax = np.array(x)
	my = np.array(y)

	mx = np.matrix([ax,np.ones(len(ax))])
	
	theta=np.multiply(inv(np.dot(mx,mx.transpose())) , np.dot(mx,my))

	return j(N, my, theta, ax)

#main
b = 5
theta1 = 2
theta2 = 5
mu, sigma = 0, 1

risque = []
n = []

#test the effect of N
print("Boucle pour tester l'effet de la valeur de N de 10 a 1000 (100 itérations) avec sigma = 1")
for N in linspace(10,1000,100):
	risque.append(compute(N,b,theta1,theta2,mu,sigma))
	n.append(N)

plt.plot(n,risque, label="f(N)=risque")
plt.legend()
plt.xlabel("n")
plt.ylabel("risque")
plt.show()

print("On peut voir que plus N augmente, plus le risque diminue car avec plus de données, on apprend plus")

#Reinit for test the effect of Sigma
N = 1000
sigmas = []
risque = []

print("Boucle pour tester l'effet de la valeur de sigma de 0.1 a 2 (100 itérations) avec N = 1000")

for sigma in linspace(0.1,2,100):
	risque.append(compute(N,b,theta1,theta2,mu,sigma))
	sigmas.append(sigma)

plt.plot(sigmas,risque, label="f(Sigma)=risque")
plt.xlabel("sigmas")
plt.ylabel("risque")
plt.legend()
plt.show()

print("On peut voir que plus la variance augmente, plus le risque augmente, car l'apprentissage est plus difficile en raison de la diversification des données") 
