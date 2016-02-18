
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

print("Initialisation de la matrice de probabilité des recompenses ...[25x25x4]")
rewards=np.zeros((sizex*sizey,sizex*sizey,4))
for y in range(sizey):
	for x in range(sizex):
		for yp in range(sizey):
			for xp in range(sizex):
				if abs(y-yp) <= 1 and abs(x-xp) <= 1:
					for a in range(4):
					#we are in range
						if y == 0 and x == 1:
							rewards[y*sizex+x][21][a] = 10
						elif y == 0 and x == 3:
							rewards[y*sizex+x][13][a] = 5
						elif (y == 0 and a == 0) or (x == sizex-1 and a == 1) or (y == sizey-1 and a == 2) or (x == 0 and a == 3):
							rewards[y*sizex+x][yp*sizex+xp][a] = -1
						else:
							rewards[y*sizex+x][yp*sizex+xp][a] = 0

rnd = 1
print("Initialisation de la matrice de probabilité des transition ...[25x25x4]")
transition=np.zeros((sizex*sizey,sizex*sizey,4))
for y in range(sizey):
	for x in range(sizex):
		if not y*sizex+x in walls:
			for a in range(4):
				new_state = y*sizex+x

				if new_state == 1:
					transition[new_state][21][a] = 1.0
				elif  new_state == 3:
					transition[new_state][13][a] = 1.0
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
					transition[y*sizex+x][new_state][a] = rnd

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
							transition[y*sizex+x][new_state_y][a] += (1-rnd)/3.0
						if not new_state_x == new_state:
							transition[y*sizex+x][new_state_x][a] += (1-rnd)/3.0

def value_iteration(transition,rewards,gamma):
	V=np.zeros((sizex*sizey))
	n = 0
	while True:
		V_p=np.zeros((sizex*sizey))
		for y in range(sizey):
			for x in range(sizex):
				s=np.zeros(4)
				for a in range(4):
					for yp in range(sizey):
						for xp in range(sizex):
							s[a] += transition[y*sizex+x][yp*sizex+xp][a] * (rewards[y*sizex+x][yp*sizex+xp][a] + gamma * V[yp*sizex+xp])
				V_p[y*sizex+x] = max(s)
		if (V==V_p).all():
			break;
		else: 	
			n = n + 1
			V = V_p

	print("Valeur optimale obtenue en " + str(n) + " iterations")
	P=np.zeros((sizex*sizey,4))	
	for y in range(sizey):
		for x in range(sizex):
			s=np.zeros(4)
			for a in range(4):
				for yp in range(sizey):
					for xp in range(sizex):
						s[a] += transition[y*sizex+x][yp*sizex+xp][a] * (rewards[y*sizex+x][yp*sizex+xp][a] + gamma * V[yp*sizex+xp])
			for a in np.where(s==max(s)):
				P[y*sizex+x][a] = 1
	return V,P

def policy_iteration(transition,rewards,gamma):
	P=np.ones((sizex*sizey,4))
	n=0
	V=np.zeros((sizex*sizey))
	while True:
		V_p=np.zeros((sizex*sizey))
		for y in range(sizey):
			for x in range(sizex):
				s=np.zeros(4)
				for a in range(4):
					for yp in range(sizey):
						for xp in range(sizex):
							s[a] += P[y*sizex+x][a] * (transition[y*sizex+x][yp*sizex+xp][a] * (rewards[y*sizex+x][yp*sizex+xp][a] + gamma * V[yp*sizex+xp]))
				V_p[y*sizex+x] = max(s)

		P_p=np.zeros((sizex*sizey,4))

		for y in range(sizey):
			for x in range(sizex):
				s=np.zeros(4)
				for a in range(4):
					for yp in range(sizey):
						for xp in range(sizex):
							s[a] += transition[y*sizex+x][yp*sizex+xp][a] * (rewards[y*sizex+x][yp*sizex+xp][a] + gamma * V_p[yp*sizex+xp])
				for a in np.where(s==max(s)):
					P_p[y*sizex+x][a] = 1
		if (P==P_p).all():
			break; 
		else: 	
			n = n + 1
			P = P_p
			V = V_p
	print("Politique optimale obtenue en " + str(n) + " iterations")
	return V,P

def print_policy(P):
	for y in range(sizey):
		line = ""
		for x in range(sizex):
			cell = ""			
			for a in range(4):
				if P[y*sizex+x][a] == 1:
					if a == 0:
						cell += "n"
					elif a == 1:
						cell += "e"
					elif a == 2:
						cell += "s"
					else:
						cell += "w"
			
			line += cell + "\t"	
		print(line)
gamma = 0.9
print("Fonction de Valeur et politique par value iteration jusqu'à convergence : Vn = Vn+1 ...")
V,P=value_iteration(transition,rewards,gamma)
print(V)
print_policy(P)
print("Fonction de Valeur et politique par policy iteration jusqu'à convergence : Pn = Pn+1 ...")
V,P=policy_iteration(transition,rewards,gamma)
print(V)
print_policy(P)


