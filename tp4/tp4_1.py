
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
	ay = np.array(y)

	mx = np.matrix([ax,np.ones(len(ax))])
	my = np.array(ay)
	theta=np.multiply(inv(np.dot(mx,mx.transpose())) , np.dot(mx,my))

	return j(N, ay, theta, ax)

#main
b = 5
theta1 = 2
theta2 = 5
mu, sigma = 0, 1

score = []
n = []

#test the effect of N
for N in linspace(10,1000,100):
	score.append(compute(N,b,theta1,theta2,mu,sigma))
	n.append(N)

plt.plot(n,score)
plt.show()

#Reinit for test the effect of Sigma
N = 1000
sigmas = []
score = []

for sigma in linspace(0.1,2,100):
	score.append(compute(N,b,theta1,theta2,mu,sigma))
	sigmas.append(sigma)

plt.plot(sigmas,score)
plt.show()
