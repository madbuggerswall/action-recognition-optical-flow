import cv2 as cv
import numpy 
import math
import matplotlib.pyplot as mpl

def isOnLeftHandSide(degree):
	if degree > math.pi/2 and degree < math.pi*3/2:
		return True
	else:
		return False

degrees = numpy.ones(shape=(13,13))
magnitudes = numpy.ones(shape=(13,13))

for i in range(degrees.shape[0]):
	for j in range(degrees.shape[1]):
		degrees[i,j] = math.radians(i*15 + j*15)

numberOfBins = 8
bins = numpy.zeros(numberOfBins)

# Mirrors the angles inside the 1st & 3rd quadrant along Y-axis
def mirrorAnglesRHS(degrees):
	for i in range(degrees.shape[0]):
		for j in range(degrees.shape[1]):
			if isOnLeftHandSide(degrees[i,j]):
				degrees[i,j] = math.pi-degrees[i,j]
			if degrees[i,j] >= math.pi*3/2:
				degrees[i,j] = degrees[i,j] - math.pi*2
	return degrees

def createHistogram(degrees, magnitudes, boundaries, bins):
	for i in range(degrees.shape[0]):
		for j in range(degrees.shape[1]):
			for b in range(bins.shape[0]):
				if(degrees[i,j] >= boundaries[b] and degrees[i,j] < boundaries[b+1]):
					bins[b] += magnitudes[i,j]
	return bins

def binBoundaries(numberOfBins):
	binValues = []
	for i in range(numberOfBins+1):
		boundary = math.radians(-90+180*i/numberOfBins)
		binValues.append(boundary)
	return binValues

# degreesRHS = mirrorAnglesRHS(degrees)
# print(degreesRHS)
# boundaries = binBoundaries(numberOfBins)
# print(boundaries)
# histogram = createHistogram(degreesRHS, magnitudes, boundaries, bins)
# print(histogram)

# hist = mpl.bar(boundaries[:numberOfBins], histogram, align="edge", width=0.3)
# mpl.show(hist)

X = numpy.arange(-10, 20, 1)
Y = numpy.arange(-10, 10, 1)
U, V = numpy.meshgrid(X, Y)

print(X.shape)
print(Y.shape)
print(U.shape)
print(V.shape)

fig, ax = mpl.subplots()
q = ax.quiver(X, Y, U, V)
ax.quiverkey(q, X=0.3, Y=1.1, U=10,
             label='Quiver key, length = 10', labelpos='E')

mpl.show()