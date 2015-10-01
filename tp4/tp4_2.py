import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import random as r

#fonction f to compute y from x
def f(x, tone, ttwo, e):
	return tone + ttwo * x + e

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

	A = np.multiply((1/pow(sigma,2)), np.dot(mx,mx.transpose()))

	tbar = np.multiply(1/pow(sigma,2), np.multiply(np.dot(inv(A).transpose(), mx), y))

b = 5
theta1 = 2
theta2 = 5
mu, sigma = 0, 1

compute(1000, b, theta1, theta2, mu, sigma)
