
import numpy as np
import random
import math
import sys

def selectAction(Q,rewards,state):
	maxQ=-sys.maxsize
	selectedAction = -1
	possibleAction = []
	
	#explore
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

#Taile du gridworld 5x5
sizex = 5
sizey = 5

#start point and goal point
start = 20
goal = 24

walls = []
walls.append(21)
walls.append(22)
walls.append(23)

print("Gridworld sarsa avec une falaise en bas du plateau")
print("4 actions possibles : NORTH, EAST, SOUTH, WEST")
print("Grille de taille 5x5 = 25")
print("Initialisation de la matrice de recompenses... [25x4]")
rewards=np.zeros((sizex*sizey,4))
for y in range(sizey):
	for x in range(sizex):
		for a in range(4):
			if (y == 0 and a == 0) or (x == sizex-1 and a == 1) or (y == sizey-1 and a == 2) or (x == 0 and a == 3):
				rewards[y*sizex+x][a] = -1
			elif y*sizex+x in walls:
				rewards[y*sizex+x][a] = -1			
			else:
				rewards[y*sizex+x][a] = 0
for a in range(4):
	rewards[goal][a] = 10

print("Initialisation de la matrice de transitions...[25x4]")
transition=np.zeros((sizex*sizey,4))
for y in range(sizey):
	for x in range(sizex):
		for a in range(4):
			if (y == 0 and a == 0) or (x == sizex-1 and a == 1) or (y == sizey-1 and a == 2) or (x == 0 and a == 3):
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
nbTurn = 10
for i in range(nbTurn):
	s = start
	a_p = selectAction(Q,rewards,s)
	s = transition[s][a_p]
	a = a_p
	r_p = rewards[s][a_p]
	while not s == goal:
	#for i in range(100):
		a_p = selectAction(Q,rewards,s)
		s_p = transition[s][a_p]
		r_p = rewards[s][a_p]
		delta = r_p + gamma*Q[s_p][a_p] - Q[s][a]
		Q[s][a] = Q[s][a] + alpha * delta
		s = s_p
		a = a_p

print(Q)
	
	

