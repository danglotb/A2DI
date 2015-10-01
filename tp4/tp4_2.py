import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import random as r

b = 5
theta1 = 2
theta2 = 5
mu, sigma = 0, 1

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
