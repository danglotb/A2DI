
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import random

def countT(Ti, t, i):
	cpt = 1
	for z in range(t):
		if Ti[z] == i:
			cpt = cpt + 1
	return cpt

def roulette(muchapeau,k):
	muchapeau_prop = []
	muchapeau_total = 0
	for i in range(len(muchapeau)):
		muchapeau_total += muchapeau[i]
	if (muchapeau_total == 0):
		return random.randint(0,k-1)
	else:
		for i in range(k):
			muchapeau_prop.append(muchapeau[i]/muchapeau_total)
		r = random.random()
		
		return 1

def softmax(k, vi, esp, n):
	Ti = []#index of the chosen at i time
	muchapeau = []
	Xi = np.zeros((n,k))#if the Ti[i] succeed
	for i in range(k):
		muchapeau.append(0)
	for i in range(n):
		Ti.append(0)
	for i in range(n):
		Ic = roulette(muchapeau,k)
		if i != 0:	
			for z in range(k):
				Xi[i][z] = Xi[i-1][z]
		if random.random() <= vi[Ic]:#success
			Xi[i][Ic] = 1 if i == 0 else Xi[i-1][Ic]+1
		muchapeau[Ic] = Xi[i][Ic] / countT(Ti,i,Ic)
	return Ti,Xi

def esp_greedy(k, vi, esp, n):
	Ti = []#index of the chosen at i time
	muchapeau = []
	Xi = np.zeros((n,k))#if the Ti[i] succeed
	for i in range(k):
		muchapeau.append(0)
	for i in range(n):
		Ti.append(0)
	for i in range(n):
		if random.random() < (1 - esp):#take argmax in muchapeau
			Ic=muchapeau.index(max(muchapeau))
		else:#take a random
			Ic=random.randint(0,k-1)
		Ti[i] = Ic
		if i != 0:	
			for z in range(k):
				Xi[i][z] = Xi[i-1][z]
		if random.random() <= vi[Ic]:#success
			Xi[i][Ic] = 1 if i == 0 else Xi[i-1][Ic]+1
		muchapeau[Ic] = Xi[i][Ic] / countT(Ti,i,Ic)
	return Ti,Xi


k=10
n=100
esp=0.5
vi = []
for i in range(k):
	vi.append(random.random())

best_arm = max(vi)

loose_at_t = np.zeros((30,n))
gain_total = np.zeros((30,n))

for run in range(30):

	ret=esp_greedy(k, vi, esp, n)
	#ret=softmax(k, vi, esp, n)

	Xi=ret[1]
	Ti=ret[0]

	for i in range(n):
		loose_at_t[run][i] = best_arm-vi[Ti[i]]
		total_at_t = 0
		for z in range(k):
			total_at_t += Xi[i][z]
		gain_total[run][i] = total_at_t

for i in range(n):
	cum_gain = 0
	cum_loose = 0
	for run in range(30):
		cum_gain += gain_total[run][i]
		cum_loose += loose_at_t[run][i]
	gain_total[0][i] = (cum_gain / 30)
	loose_at_t[0][i] = (cum_loose / 30)

x=np.arange(0,n)
plt.plot(x,np.cumsum(loose_at_t[0]))
plt.plot(x,gain_total[0])
plt.show()




