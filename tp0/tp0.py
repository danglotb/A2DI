"""
Simple demo with multiple subplots.
"""
import numpy as np
import matplotlib.pyplot as plt

def plotFile(x, y):
	fx = open(x, 'r')
	fy = open(y, 'r')

	ax = []
	ay = []

	for line in fx:
		ax.append(line)
	for line in fy:
		ay.append(line)

	ax = np.array(ax)
	ay = np.array(ay)

	plt.plot(ax,ay,'bs')
	plt.show()


