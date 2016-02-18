
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from pylab import ylim
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
			elif y*sizex+x in world.cliff:
				line += "X"
			else:
				line += " "
		print(line)

def selectionAction(Q,state,history,transition):
	maxQ=-sys.maxsize
	selectedAction = -1
	possibleAction = []

	#select the best action
	for action in range(4):
		if Q[state][action] > maxQ and not transition[state][action] in history:
			maxQ = Q[state][action]
			selectedAction = action
			possibleAction = []
			possibleAction.append(action)
		elif Q[state][action] == maxQ and not transition[state][action] in history:
			possibleAction.append(action)
			selectedAction = -1
	if not selectedAction == -1:
		return selectedAction
	else:
		if len(possibleAction) > 0:
			return possibleAction[random.randint(0,len(possibleAction)-1)]
		else:
			return random.randint(0,3)


#in order to build the best path computed
def solution(Q):
	s=start
	history = []
	while not s == world.goal:
		history.append(s)
		s=transition[s][selectionAction(Q,s,history,world.transition)]
	history.append(s)
	printSolution(history)

#parametres learning
alpha=0.8
gamma=0.05

print("Paramètres d'apprentissage : "+ str(alpha))

#init the world
sizex,sizey,start,goal,cliff,transition,rewards=world.init()
nbTrial=500 if len(sys.argv) <= 1 else int(sys.argv[1])
s_rwd,Q_sarsa=learning.sarsa(nbTrial,gamma,alpha)
q_rwd,Q_Qlearning=learning.QLearning(nbTrial,gamma,alpha)

print("Solution SARSA")
solution(Q_sarsa)
print("Solution QLearning")
solution(Q_Qlearning)

ylim(-20,10)
plt.plot(range(nbTrial), s_rwd, label="SARSA")
plt.plot(range(nbTrial), q_rwd, label="Qlearning")
plt.xlabel("trials")
plt.ylabel("rwd/trials")
plt.legend()
plt.show()

print("On peut voir sur le graphique que QLearning est souvent en dessous de SARSA")
print("En revanche Qlearning prend le chemin optimal tandis que SARSA s'éloigne de la cliff")
print("pour prendre la chemin le plus sûr.")
print("Parfois, SARSA obtient un résultat très mauvais, il parcours le labyrinthe dans tout les sens.(necessite peut-etre de relancer le script)")
