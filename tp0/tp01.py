import numpy as np
from pylab import *
import matplotlib.pyplot as plt

def fitness(predicted, measured, coord_x):
	acc = 0	
	for i in range(len(predicted)):
		acc+= math.sqrt(pow(predicted[i]-measured[i],2))
	print(acc/len(predicted))

def crossValid(ax, ay, begT, endT, testData, testValue):
	mx = np.matrix([ax[begT:endT],np.ones(endT-begT)])
	my = np.array(ay[begT:endT])
	theta=np.multiply(inv(np.dot(mx,mx.transpose())),np.dot(mx,my))
	f=np.linspace(min(ax[begT:endT]), max(ax[begT:endT]))
	#data
	nax = np.array(ax[begT:endT])
	nay = np.array(ay[begT:endT])
	plt.plot(nax,nay,'bs')
	plt.plot(f,np.add(np.multiply(f,theta[0,0]),theta[1,1]))

	#test
	naxd = np.array(testData)
	nayd = np.array(testValue)
	naym = []
	for i in range(len(testData)):
		naym.append(testData[i]*theta[0,0]+theta[1,1])
	naym = np.array(naym)
	fitness(naym,nayd,naxd)
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
crossValid(ax,ay,0,80,ax[80:100],ay[80:100])
crossValid(ax,ay,10,90,ax[0:10]+ax[90:100],ay[0:10]+ay[90:100])
crossValid(ax,ay,20,80,ax[0:20],ay[0:20])




	
