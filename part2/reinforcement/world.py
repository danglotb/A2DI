import numpy as np

#Taile du gridworld 12x4
sizex = 12
sizey = 4

#start point and goal point
start = 36
goal = 47

#falaise du bord du tableaux
cliff = []

#tableaux de récompenses
rewards=np.zeros((sizex*sizey,4))
transition=np.zeros((sizex*sizey,4))

def init():

	print("Gridworld sarsa avec une falaise en bas du plateau")

	for i in range (37,47,1):
		cliff.append(i)

	print("4 actions possibles : NORTH, EAST, SOUTH, WEST")
	print("Grille de taille 12x4 = 48")

	print("Initialisation de la matrice de recompenses... [48x4]")
	print("-100 pour sortir de la falaise, -1 à chaque mouvement.")

	for y in range(sizey):
		for x in range(sizex):
			for a in range(4):
				if y*sizex+x in cliff:
					rewards[y*sizex+x][a] = -100
				else:
					rewards[y*sizex+x][a] = -1		
			

	print("Initialisation de la matrice de transitions...[48x4]")
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

	return sizex,sizey,start,goal,cliff,transition,rewards

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
