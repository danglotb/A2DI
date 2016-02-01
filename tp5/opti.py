
import numpy as np
import random
import math
import sys

#Taile du gridworld 5x5
sizex = 5
sizey = 5

#politique aléatoire : 0.25 pour chaque action
prob_action = 0.25

walls = []
if (len(sys.argv) > 1):
	while len(walls) < int(sys.argv[1]):
		rnd = random.randint(0,sizex*sizey-1)
		if not rnd in walls and rnd != 1 and rnd != 3:
			walls.append(rnd)
	print("Indices des "+str((sys.argv[1]))+" murs générés : "+ str(walls))
	for y in range(sizey):
		line = "" 
		for x in range(sizex):
			if y*sizex+x in walls:
				line += 'W'
			else:
				line += ' '
		print(line)



print("Gridworld : Calcul de la politique optimale")
print("4 actions possibles : NORTH, EAST, SOUTH, WEST")
print("Grille de taille 5x5 = 25")
print("Case A (+10) en (0,1) -> (4,1)")
print("Case B (+5)  en (0,3) -> (2,3)")
print("Initialisation de la matrice de récompenses... [25x4]")
rnd = 0.85
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

print("Initialisation de la matrice de probabilité des transition (85% - 15%)...[25x25x4]")
proba_transition=np.zeros((sizex*sizey,sizex*sizey,4))
for y in range(sizey):
	for x in range(sizex):
		if not y*sizex+x in walls:
			for a in range(4):
				new_state = y*sizex+x

				if new_state == 1:
					proba_transition[new_state][21][a] = 1.0
				elif  new_state == 3:
					proba_transition[new_state][13][a] = 1.0
				else:
					if (y == 0 and a == 0) or (x == sizex-1 and a == 1) or (y == sizey-1 and a == 2) or (x == 0 and a == 3):
						new_state = y*sizex+x
					elif a == 0:
						new_state = ((y-1)*sizex)+x
					elif a == 1:
						new_state = y*sizex+x+1
					elif a == 2:
						new_state = ((y+1)*sizex)+x
					else:
						new_state = y*sizex+x-1
					proba_transition[y*sizex+x][new_state][a] = rnd

					for v in [-1,1]:
						new_state_y = ((y+v)*sizex)+x
						if new_state_y >= sizex*sizey or new_state_y < 0:
							new_state_y = (y*sizex)+x
						new_state_x = (y*sizex)+x+v
						if new_state_x >= sizex*sizey or new_state_x < 0:
							new_state_x = (y*sizex)+x
						if new_state_y in walls:	
							new_state_y = y*sizex+x
						if new_state_x in walls:	
							new_state_x = y*sizex+x
						if not new_state_y == new_state:
							proba_transition[y*sizex+x][new_state_y][a] += (1-rnd)/3.0
						if not new_state_x == new_state:
							proba_transition[y*sizex+x][new_state_x][a] += (1-rnd)/3.0


print("Calcul de R_pi...[25]")
R_pi=np.zeros(sizey*sizex)
for y in range(sizey):
	for x in range(sizex):
		s=0
		for a in range(4):
			s += prob_action * rewards[y*sizex+x][a]
		R_pi[y*sizex+x]=s

print("Calcul de P_pi...[25x25]")
P_pi=np.zeros((sizey*sizex,sizey*sizex))
for y_pi in range(sizey):
	for x_pi in range(sizex):	
		for y in range(sizey):
			for x in range(sizex):
				for a in range(4):
					P_pi[y_pi*sizex+x_pi][y*sizex+x] += prob_action * proba_transition[y_pi*sizex+x_pi][y*sizex+x][a]

print("Calcul de la fonction de Valeur = (I - (0.9 * P_pi))^-1 * R_pi )")
#(I - (Gamma * P_pi))^-1 * R_pi
gamma=0.9
V = np.dot(np.linalg.inv(np.subtract(np.identity(sizey*sizex), np.multiply(gamma, P_pi))), R_pi)
for y in range(sizey):
	line = "" 
	for x in range(sizex):
		line += str(V[y*sizex+x]) + "\t"
	print(line)
