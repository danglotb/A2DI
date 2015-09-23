"""
Simple demo with multiple subplots.
"""
import numpy as np
from pylab import *
import matplotlib.pyplot as plt

x='x.txt'
y='y.txt'

fx = open(x, 'r')
fy = open(y, 'r')

ax = []
ay = []

for line in fx:
	ax.append(float(line))
for line in fy:
	ay.append(float(line))

mx = np.matrix([ax,np.ones(len(ax))])
my = np.array(ay)

np.multiply( inv(np.dot(mx,mx.transpose())) , np.dot(mx,my))

ax = np.array(ax)
ay = np.array(ay)
plt.plot(ax,ay,'bs')
plt.show()


