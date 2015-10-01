import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import cross_validation

k = 10

x = 'x2.txt'
y = 'y2.txt'

fx = open(x, 'r')
fy = open(y, 'r')

ax = []
ay = []

for line in fx:
	ax.append(float(line))
for line in fy:
	ay.append(float(line))

x = np.array(ax)[:, np.newaxis]
y = np.array(ay)

x_plot = np.array(ax)[:, np.newaxis]

bestScore = -1

kf = cross_validation.KFold(len(ax),k,shuffle=True)

#using SciKit to do the crossValidation + polynomial interpolation
#it will keep the best degree (function and plot) to display it after
for train, test in kf:
	xTrain = x[train]
	yTrain = y[train]
	xTest = x[test]
	yTest = y[test]
	for degree in range(1, 24):
		model = make_pipeline(PolynomialFeatures(degree), Ridge())
		model.fit(xTrain, yTrain)
		tmpScore = model.score(xTest,yTest)
		if tmpScore > bestScore:
			bestScore = tmpScore
			y_plotBest = model.predict(x_plot)
			bestDegree = degree

plt.plot(x_plot, y_plotBest, label="degree %d" % bestDegree)

nax = np.array(ax)
nay = np.array(ay)
plt.plot(nax,nay,'bs')

plt.legend(loc='lower left')
plt.show()

fx.close()
fy.close()
