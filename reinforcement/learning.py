
import numpy as np
import random
import math
#from matplotlib import pyplot as plt
import sys
import world

epsilon=0.8

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
		return possibleAction[random.randint(0,len(possibleAction)-1)]

def sarsa(nbTrial,gamma,alpha,debug=False):
	print("Sarsa learning")
	array_rewards=[]
	Q=np.ones((world.sizex*world.sizey, 4))
	for i in range(nbTrial):
		if debug:
			print(i)
		s = world.start
		a = selectAction(Q,s)
		while not s == world.goal:
			r = world.rewards[s][a]
			array_rewards.append(r)
			s_p = world.transition[s][a]
			a_p = selectAction(Q,s_p)
			Q[s][a] = Q[s][a] + (alpha * (r + gamma*Q[s_p][a_p] - Q[s][a]))
			s = s_p
			a = a_p
	return (sum(array_rewards)/nbTrial),Q

def QLearning(nbTrial,gamma,alpha,debug=False):
	print("QLearning")
	array_rewards=[]
	Q=np.ones((world.sizex*world.sizey, 4))
	for i in range(nbTrial):
		if debug:
				print(i)
		s = world.start
		while not s == world.goal:
			a = selectAction(Q,s)
			s_p = world.transition[s][a]
			r = world.rewards[s][a]
			array_rewards.append(r)
			Q[s][a] = Q[s][a] + (alpha * (r + gamma*Q[s_p][a] - Q[s][a]))
			s = s_p
	return (sum(array_rewards)/nbTrial),Q
