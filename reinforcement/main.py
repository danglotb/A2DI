
import numpy as np
import random
import math
#from matplotlib import pyplot as plt
import sys
import world
import learning

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

def selectionAction(Q,state,history):
	return int(np.where(Q[state]==max(Q[state]))[0])


#in order to build the best path computed
def solution(Q):
	s=start
	history = []
	while not s == world.goal:
		history.append(s)
		s=transition[s][int(np.where(Q[s] == max(Q[s]))[0])]
	history.append(s)
	printSolution(history)

#parametres learning
alpha=0.9
gamma=0.1

#init the world
sizex,sizey,start,goal,cliff,transition,rewards=world.init()

sarsa_rwd,Q_sarsa=learning.sarsa(100,gamma,alpha)
QLearning_rwd,Q=learning.QLearning(100,gamma,alpha)
print("Solution Sarsa")
solution(Q_sarsa)
print("Solution QLearning")
solution(Q)

'''
sarsa_rwd = []
QLearning_rwd = []
for i in range(s,1000,25):
	print(i)
	sarsa_rwd.append(learning.sarsa(100,gamma,alpha)[0])
	QLearning_rwd.append(QLearning(i,gamma,alpha)[0])


plt.plot(sarsa_rwd,range(25,1000,25), label="sarsa")
plt.plot(QLearning_rwd,range(25,1000,25), label="sarsa")
plt.xlabel("trials")
plt.ylabel("rwd/trials")
plt.legend()
plt.show()
'''
