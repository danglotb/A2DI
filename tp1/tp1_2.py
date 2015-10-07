import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import random as r

#fonction f to compute y from x
def f(x, tone, ttwo, e):
	return tone + ttwo * x + e

def normal(x,mu,sigma):
	return np.exp(np.multiply((-1/2),pow((np.subtract(x,mu)/sigma),2)))

#compute
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

	SigmaM =  np.multiply(pow(sigma,2),np.identity(2))
	A = inv(np.dot( np.multiply((1/pow(sigma,2)), mx), mx.transpose()) + inv(SigmaM))
	tbar = np.dot(np.dot(np.multiply(1/pow(sigma,2), A), mx), my)
	
	p =  np.random.normal(tbar,A)

b = 5
theta1 = 2
theta2 = 5
mu, sigma = 0, 1

compute(1000, b, theta1, theta2, mu, sigma)
