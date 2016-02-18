
import numpy as np
import random
import math
#from matplotlib import pyplot as plt
import sys
import world
epsilon=0.85
#Function E-greedy which return the best action
def selectAction(Q,state,explore=True):
	maxQ=-sys.maxsize
	selectedAction = -1
	possibleAction = []

	#explore
	if explore:
		if random.random() > epsilon:
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
		if (len(possibleAction) > 0):
			return possibleAction[random.randint(0,len(possibleAction)-1)]
		else:
			return random.randint(0,3)

def sarsa(nbTrial,gamma,alpha,debug=False):
	print("Sarsa learning avec " + str(nbTrial) + " iterations...")
	rewards = []
	Q=np.zeros((world.sizex*world.sizey, 4))
	for _ in range(nbTrial):
		reward = 0
		cpt = 0
		if debug:
			print(i)
		s = world.start
		a = selectAction(Q,s)
		while not s == world.goal:
			r = world.rewards[s][a]
			reward += r
			s_p = world.transition[s][a]
			a_p = selectAction(Q,s_p)
			Q[s][a] = Q[s][a] + (alpha * (r + gamma*Q[s_p][a_p] - Q[s][a]))
			s = s_p
			a = a_p
			cpt += 1
		rewards.append(reward/cpt)
	return rewards,Q

def QLearning(nbTrial,gamma,alpha,debug=False):
	print("QLearning avec " + str(nbTrial) + " iterations...")
	rewards = []
	Q=np.zeros((world.sizex*world.sizey, 4))
	for _ in range(nbTrial):
		reward = 0
		cpt = 0
		if debug:
			print(i)
		s = world.start
		while not s == world.goal:
			a = selectAction(Q,s)
			s_p = world.transition[s][a]
			r = world.rewards[s][a]
			reward += r
			Q[s][a] = Q[s][a] + (alpha * (r + gamma* max(Q[s_p]) - Q[s][a]))
			s = s_p
			cpt += 1
		rewards.append(reward/cpt)
	return rewards,Q
