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

theta=np.multiply(inv(np.dot(mx,mx.transpose())) , np.dot(mx,my))

nax = np.array(ax)
nay = np.array(ay)
plt.plot(nax,nay,'bs')

f=np.linspace(min(ax), max(ax))

plt.plot(f,np.add(np.multiply(f,theta[0,0]) , theta[1,1]))

plt.show()
