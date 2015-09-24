import numpy as np
from pylab import *
import matplotlib.pyplot as plt

x='x2.txt'
y='y2.txt'

fx = open(x, 'r')
fy = open(y, 'r')

ax = []
ay = []

for line in fx:
	ax.append(float(line))
for line in fy:
	ay.append(float(line))

nax = np.array(ax)
nay = np.array(ay)
plt.plot(nax,nay,'bs')

plt.show()
