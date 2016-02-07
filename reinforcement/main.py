
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import sys

#to display the world
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

#to display the solution
def printSolution(history):
	for y in range(sizey):
		line = ""
		for x in range(sizex):
			if y*sizex+x in history:
				line += "."
			else:
				line += " "
		print(line)

#in order to build the best path computed
def solution(Q,state,history,transition):
	maxQ=-sys.maxsize
	selectedAction = -1
	possibleAction = []

	for action in range(4):
		if Q[state][action] > maxQ:
			if not transition[state][action] in history:
				maxQ = Q[state][action]
				selectedAction = action
				possibleAction = []
				possibleAction.append(action)
		elif Q[state][action] == maxQ:
			if not transition[state][action] in history:
				possibleAction.append(action)
				selectedAction = -1
	if not selectedAction == -1:
		return selectedAction
	else:
		if len(possibleAction) <= 0:
			print(Q[state])
			print(max(Q[state]))
			return -1
		else:
			return possibleAction[random.randint(0,len(possibleAction)-1)]


#Function E-greedy which return the best action
def selectAction(Q,state,explore=True):
	maxQ=-sys.maxsize
	selectedAction = -1
	possibleAction = []

	#explore
	if explore:
		if random.random() > 0.90:
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

def sarsa(nbTrial):
	print("Sarsa learning")
	array_rewards=[]
	Q=np.ones((sizex*sizey, 4))
	for _ in range(nbTrial):
		s = start
		a = selectAction(Q,s)
		while not s == goal:
			r = rewards[s][a]
			array_rewards.append(r)
			s_p = transition[s][a]
			a_p = selectAction(Q,s_p)
			Q[s][a] = Q[s][a] + (alpha * (r + gamma*Q[s_p][a_p] - Q[s][a]))
			s = s_p
			a = a_p
	return (sum(array_rewards)/nbTrial),Q

def QLearning(nbTrial):
	print("QLearning")
	array_rewards=[]
	Q=np.ones((sizex*sizey, 4))
	for _ in range(nbTrial):
		s = start
		while not s == goal:
			a = selectAction(Q,s)
			s_p = transition[s][a]
			r = rewards[s][a]
			array_rewards.append(r)
			Q[s][a] = Q[s][a] + (alpha * (r + gamma*Q[s_p][a] - Q[s][a]))
			s = s_p
	return (sum(array_rewards)/nbTrial),Q

#parametres learning
alpha = 0.9
gamma=0.1

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
			elif y==0:
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
sarsa_rwd = []
QLearning_rwd = []
for i in range(25,1000,25):
	print(i)
	sarsa_rwd.append(sarsa(50)[0])
	QLearning_rwd.append(QLearning(50)[0])

plt.plot(sarsa_rwd,range(25,1000,25), label="sarsa")
plt.xlabel("trials")
plt.ylabel("rwd/trials")
plt.legend()
plt.show()
