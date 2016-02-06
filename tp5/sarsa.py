
import numpy as np
import random
import math
import sys

def printGW():
	for y in range(sizey):
		line = ""
		for x in range(sizex):
			if y*sizex+x in cliff:
				line += "X"
			elif y*sizex+x == start:
				line += "S"
			elif y*sizex+x == goal:
				line += "G"
			else:
				line += " "
		print(line)

def selectAction(Q,state,explore=True,action=None):
	maxQ=-sys.maxsize
	selectedAction = -1
	possibleAction = []
	
	#explore
	if explore:
		if random.random() > 0.80:
			return random.randint(0,3)

	#select the best action
	for action in range(4):
		if Q[state][action] > maxQ:
			maxQ = Q[state][action]
			selectedAction = action
			possibleAction = []
			possibleAction.append(action)
		elif Q[state][action] == maxQ:
			possibleAction.append(action)
			selectedAction = -1
	if not selectedAction == -1:
		return selectedAction
	else:
		return possibleAction[random.randint(0,len(possibleAction)-1)]

#parametres learning
alpha=0.85
gamma=0.05

#Taile du gridworld 12x4
sizex = 12
sizey = 4

#start point and goal point
start = 36
goal = 47

cliff = []

for i in range (37,47,1):
	cliff.append(i)


print("Gridworld sarsa avec une falaise en bas du plateau")
print("4 actions possibles : NORTH, EAST, SOUTH, WEST")
print("Grille de taille 12x4 = 48")
print("Initialisation de la matrice de recompenses... [48x4]")
rewards=np.zeros((sizex*sizey,4))
for y in range(sizey):
	for x in range(sizex):
		for a in range(4):
			if y*sizex+x in cliff:
				rewards[y*sizex+x][a] = -100
			elif (y == 0 and a == 0) or (x == sizex-1 and a == 1) or (y == sizey-1 and a == 2) or (x == 0 and a == 3):
				rewards[y*sizex+x][a] = -1
			else:
				rewards[y*sizex+x][a] = 0

print("Initialisation de la matrice de transitions...[48x4]")
transition=np.zeros((sizex*sizey,4))
for y in range(sizey):
	for x in range(sizex):
		for a in range(4):
			if y*sizex+x in cliff:
				transition[y*sizex+x][a] = start
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



#Init Q_0
print("Sarsa learning")
Q=np.zeros((sizex*sizey, 4))
nbTurn = 1

for _ in range(100):
	s = start
	a = selectAction(Q,s)
	s = transition[s][a]
	r = rewards[s][a]
	while not s == goal:
		a_p = selectAction(Q,s)
		r_p = rewards[s][a_p]
		s_p = transition[s][a_p]
		delta = r_p + gamma*Q[s_p][a_p] - Q[s][a]
		Q[s][a] = Q[s][a] + (alpha * delta)
		s = s_p
		a = a_p

print(Q[start])
'''
print("Solution ...")
s = start
while not s == goal:
	a = selectAction(Q,s,explore=False)
	s = transition[s][a]
	print(a)'''		

