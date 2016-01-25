
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import random
import math

from enum import Enum

class action(Enum):
	north = 0
	east = 1
	south = 2
	west = 3

sizex = 5
sizey = 5

prob_action = 0.25

rewards=np.zeros((sizex*sizey,4))
for y in range(sizey):
	for x in range(sizex):
		for a in range(4):
			if y == 0 and x == 1:
				rewards[y*sizex+x][a] = 10
			elif y == 0 and x == 3:
				rewards[y*sizex+x][a] = 5
			elif (y == 0 and a == 0) or (x == sizex-1 and a == 1) or (y == sizey-1 and a == 2) or (x == 0 and a == 3):
				rewards[y*sizex+x][a] = -1
			else:
				rewards[y*sizex+x][a] = 0

transition=np.zeros((sizex*sizey,4))
for y in range(sizey):
	for x in range(sizex):
		for a in range(4):
			if y == 0 and x == 1:
				transition[y*sizex+x][a] = 21
			elif y == 0 and x == 3:
				transition[y*sizex+x][a] = 13
			elif (y == 0 and a == 0) or (x == sizex-1 and a == 1) or (y == sizey-1 and a == 2) or (x == 0 and a == 3):
				transition[y*sizex+x][a] = y*sizex+x
			elif a == 0:
				transition[y*sizex+x][a] = ((y-1)*sizex)+x
			elif a == 1:
				transition[y*sizex+x][a] = y*sizex+x+1
			elif a == 2:
				transition[y*sizex+x][a] = ((y+1)*sizex)+x
			else:
				transition[y*sizex+x][a] = y*sizex+x-1


proba_res=1.0

R_pi=np.zeros(sizey*sizex)
for y in range(sizey):
	for x in range(sizex):
		s=0
		for a in range(4):
			s += prob_action * rewards[y*sizex+x][a]
		R_pi[y*sizex+x]=s


P_pi=np.zeros((sizey*sizex,sizey*sizex))
for y_pi in range(sizey):
	for x_pi in range(sizex):
		for y in range(sizey):
			for x in range(sizex):
				for a in range(4):
					P_pi[y_pi*sizex+x_pi][y*sizex+x] += prob_action if transition[y_pi*sizex+x_pi][a] == y*sizex+x else 0


#(I - (Gamma * P_pi))^-1 * R_pi
gamma=0.9
V = np.dot(inv(np.subtract(np.identity(sizey*sizex), np.multiply(gamma, P_pi))), R_pi)

print(V)
