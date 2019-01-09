import cv2 as cv
import numpy 
import math
import matplotlib.pyplot as mpl

#	Checks if the given angle is inside the 1st or 3rd quadrant.
def isOnLeftHandSide(degree):
	if degree > math.pi/2 and degree < math.pi*3/2:
		return True
	else:
		return False

# Mirrors the angles inside the 1st & 3rd quadrant along Y-axis
def mirrorAnglesRHS(degrees):
	for i in range(degrees.shape[0]):
		for j in range(degrees.shape[1]):
			if isOnLeftHandSide(degrees[i,j]):
				degrees[i,j] = math.pi-degrees[i,j]
			if degrees[i,j] >= math.pi*3/2:
				degrees[i,j] = degrees[i,j] - math.pi*2
	return degrees

# Returns a numpy.ndarray
def createHistogram(degrees, magnitudes, boundaries, bins):
	for i in range(degrees.shape[0]):
		for j in range(degrees.shape[1]):
			for b in range(bins.shape[0]):
				if(degrees[i,j] >= boundaries[b] and degrees[i,j] < boundaries[b+1]):
					bins[b] += magnitudes[i,j]
	return bins

# Define bin boundaries according to number of bins.
def binBoundaries(numberOfBins):
	binValues = []
	for i in range(numberOfBins+1):
		boundary = math.radians(-90+180*i/numberOfBins)
		binValues.append(boundary)
	return binValues

