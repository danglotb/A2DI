
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import random
import math

def countT(Ti, t, i):
	cpt = 1
	for z in range(t):
		if Ti[z] == i:
			cpt = cpt + 1
	return cpt

#Not used
def roulette(muchapeau,k):#doesn't work 
	muchapeau_prop = []
	muchapeau_total = 0
	for i in range(len(muchapeau)):
		muchapeau_total += muchapeau[i]
	if (muchapeau_total == 0):
		return random.randint(0,k-1)
	else:
		cu_muchapeau = 0.0
		for i in range(k):
			if cu_muchapeau + muchapeau[i] / muchapeau_total >= r:
				return i
			cu_muchapeau += muchapeau[i] / muchapeau_total
		return -1

def boltzmann(muchapeau,k,To):
	cu_muchapeau = 0.0
	muchapeau_total = 0
	r = random.random()
	for i in range(len(muchapeau)):
		muchapeau_total += exp(muchapeau[i]/To)
	for i in range(k):
		if cu_muchapeau + exp(muchapeau[i]/To)/muchapeau_total >= r:
			return i
		cu_muchapeau += exp(muchapeau[i]/To)/muchapeau_total 
	return -1

def softmax(k, vi, n, To):
	Ti = []#index of the chosen at i time
	for i in range(n):
		Ti.append(0)
	muchapeau = np.zeros(k)
	Xi = np.zeros((n,k))#if the Ti[i] succeed
	for i in range(n):
		Ic = boltzmann(muchapeau,k,To)	
		if i != 0:	
			for z in range(k):
				Xi[i][z] = Xi[i-1][z]
		if random.random() <= vi[Ic]:#success
			Xi[i][Ic] = 1 if i == 0 else Xi[i-1][Ic]+1
		muchapeau[Ic] = Xi[i][Ic] / countT(Ti,i,Ic)
		Ti[i] = Ic
	return Ti,Xi

def beta_law(a,b,v):
	alpha = 2
	return alpha * pow(v, a-1) * pow( (1-v), b-1)#zz a revoir

def thompson_sampling(k,n,vi):
	alpha = 1
	beta = 1
	s = np.zeros(k)
	e = np.zeros(k)
	array = []
	Ti = []#index of the chosen at i time
	for i in range(n):
		Ti.append(0)
	Xi = np.zeros((n,k))#if the Ti[i] succeed
	for i in range(n):
		for z in range(k):
			array.append(beta_law(s[z]+alpha,e[z]+beta,s[z]/i if i != 0 else 0.5))
		Ti[i]= (np.where(array==np.amax(array)))[0][0]
		if i != 0:
			for z in range(k):
				Xi[i][z] = Xi[i-1][z]
		if random.random() <= vi[Ti[i]]:#success
			e[Ti[i]] += 1
			Xi[i][Ti[i]] = 1 if i == 0 else Xi[i-1][Ti[i]]+1
		else:
			s[Ti[i]] += 1
		array.clear()
	return Ti,Xi

def ucb(k,n,vi):
	Ti = []
	for i in range(n):
		Ti.append(0)
	muchapeau = np.zeros(k)
	Xi = np.zeros((n,k))
	array = []
	alpha = 2
	for i in range(n): 
		for z in range(k):
			array.append(muchapeau[z]+math.sqrt( (alpha*math.log(i+1)) / countT(Ti,i,z)))
		Ic = (np.where(array==np.amax(array)))[0][0]	
		Ti[i] = Ic
		if i != 0:
			for z in range(k):
				Xi[i][z] = Xi[i-1][z]
		if random.random() <= vi[Ic]:#success
			Xi[i][Ic] = 1 if i == 0 else Xi[i-1][Ic]+1
		muchapeau[Ic] = Xi[i][Ic] / countT(Ti,i,Ic)
		array.clear()
	return Ti,Xi
	

def esp_greedy(k, vi, n):
	esp = 0.5
	Ti = []
	for i in range(n):
		Ti.append(0)
	muchapeau = np.zeros(k)
	Xi = np.zeros((n,k))#if the Ti[i] succeed
	for i in range(n):
		if random.random() < (1 - esp):#take argmax in muchapeau
			Ic=np.where(muchapeau==max(muchapeau))[0][0]
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


k=10#nombre de bras
n=100#temps
nbrun=30#nombre de repetition de l'experience
vi = []#distribution de probabilité de reussite pour chaque bras
for i in range(k):
	vi.append(random.random())

best_arm = max(vi)#recuperation du "meilleur" bras pour calculer le regret (apres experience)

loose_at_t = np.zeros((nbrun,n))
gain_total = np.zeros((nbrun,n))

for run in range(nbrun):

	#ret=esp_greedy(k, vi, n)
	#ret=softmax(k, vi, n)doesn't work
	#ret=softmax(k, vi, n, To = 0.1)
	ret=thompson_sampling(k,n,vi)
	#ret=ucb(k,n,vi)

	Xi=ret[1]
	Ti=ret[0]

	for i in range(n):
		loose_at_t[run][i] = best_arm-vi[Ti[i]]
		total_at_t = 0
		for z in range(k):
			total_at_t += Xi[i][z]
		gain_total[run][i] = total_at_t

#calcul des moyennes de gain et de regret
for i in range(n):
	cum_gain = 0
	cum_loose = 0
	for run in range(nbrun):
		cum_gain += gain_total[run][i]
		cum_loose += loose_at_t[run][i]
	gain_total[0][i] = (cum_gain / nbrun)
	loose_at_t[0][i] = (cum_loose / nbrun)

x=np.arange(0,n)
plt.plot(x,np.cumsum(loose_at_t[0]), label='regret')
plt.plot(x,gain_total[0],label='gain')
plt.legend(loc='lower right')
plt.show()




