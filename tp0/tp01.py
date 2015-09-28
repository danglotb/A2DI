import numpy as np
from pylab import *
import matplotlib.pyplot as plt

score = []
thetas = []

def fitness(predicted, measured, coord_x):
	acc = 0	
	for i in range(len(predicted)):
		acc+= math.sqrt(pow(predicted[i]-measured[i],2))
	score.append(acc/len(predicted))

def crossValidation(ax,ay,index, b = False):
	for i in range(5):
		begT = i*20
		endT = (i+1)*20
		x = []
		y = []
		for j in range(100):
			if j not in index[begT:endT]:
				x.append(ax[j])
				y.append(ay[j])

		mx = np.matrix([x,np.ones(len(x))])
		my = np.array(y)
		theta=np.multiply(inv(np.dot(mx,mx.transpose())),np.dot(mx,my))
		f=np.linspace(min(x), max(x))

		nax = np.array(x)
		nay = np.array(y)
		thetas.append(theta)
		if b:	
			plt.plot(nax,nay,'bs')
			plt.plot(f,np.add(np.multiply(f,theta[0,0]),theta[1,1]))
		
		x = []
		y = []
		for j in range(100):
			if j in index[begT:endT]:
				x.append(ax[j])
				y.append(ay[j])
		naxd = np.array(x)
		nayd = np.array(y)
		naym = []
		for i in range(len(x)):
			naym.append(x[i]*theta[0,0]+theta[1,1])
		naym = np.array(naym)
		fitness(naym,nayd,naxd)
		if b:
			plt.plot(naxd,nayd,'gs')
			plt.plot(naxd,naym)
			plt.show()

def learnModel(ax,ay):
	mx = np.matrix([ax,np.ones(len(ax))])
	my = np.array(ay)
	theta=np.multiply(inv(np.dot(mx,mx.transpose())) , np.dot(mx,my))
	f=np.linspace(min(ax), max(ax))
	#fun ax+b
	plt.plot(f,np.add(np.multiply(f,theta[0,0]),theta[1,1]))
	#data
	nax = np.array(ax)
	nay = np.array(ay)
	plt.plot(nax,nay,'bs')
	plt.show()

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

#1rst learn the model
#learnModel(ax,ay)
#2nd cross validation
index = np.arange(100)
shuffle(index)
crossValidation(ax,ay,index)
print(str(thetas[score.index(min(score))][0,0])+"x+"+str(thetas[score.index(min(score))][1,1]))
print(min(score))

