
import random

import numpy as np

def countT(Ti, t, i):
	cpt = 1
	for z in range(t):
		if Ti[z] == i:
			cpt = cpt + 1
	return cpt

def esp_greedy(k, vi, esp, n):
	Ti = []#index of the chosen at i time
	muchapeau = []
	Xi = np.zeros((n,k))#if the Ti[i] succeed
	for i in range(k):
		muchapeau.append(0)
	for i in range(n):
		Ti.append(0)
	for i in range(n):
		if random.random() < 1 - esp:#take argmax in muchapeau
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

ret=esp_greedy(k,vi,esp,n)

best_arm_at_t = []

for i in range(n):
	best_arm_at_t.append(np.argmax(ret[1][i]))
