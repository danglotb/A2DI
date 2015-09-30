import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import random as r

def fitness(predicted, measured, coord_x):
	acc = 0	
	for i in range(len(predicted)):
		acc+= math.sqrt(pow(predicted[i]-measured[i],2))
	print(acc/len(predicted))

def f(x, tone, ttwo, e):
	return tone + ttwo * x + e

N = 1000
b = 5
theta1 = 2
theta2 = 5
mu, sigma = 0, 1



x = []
y = []

while len(x) != N:
	rand = r.random()*b - b/2
	x.append(rand)
	y.append(f(rand, theta1, theta2, np.random.normal(mu, sigma)))

ax = np.array(x)
ay = np.array(y)
plt.plot(ax,ay,'bs')

mx = np.matrix([ax,np.ones(len(ax))])
my = np.array(ay)
theta=np.multiply(inv(np.dot(mx,mx.transpose())) , np.dot(mx,my))
f=np.linspace(min(ax), max(ax))
plt.plot(f,np.add(np.multiply(f,theta[0,0]),theta[1,1]))

naxd = np.array(x)
nayd = np.array(y)
naym = []
for i in range(len(x)):
	naym.append(x[i]*theta[0,0]+theta[1,1])
naym = np.array(naym)
fitness(naym,nayd,naxd)

print(theta1)
print(theta[1,1])

print(theta2)
print(theta[0,0])

plt.show()

