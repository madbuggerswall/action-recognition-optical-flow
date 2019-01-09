import math
import os

import cv2 as cv
import numpy as numpy
import matplotlib.pyplot as mpl

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#	Checks if the given angle is inside the 1st or 3rd quadrant.
def isOnLeftHandSide(degree):
	if degree > math.pi/2 and degree < math.pi*3/2:
		return True
	else:
		return False

# Mirrors the angles inside the 1st & 3rd quadrant along Y-axis
def mirrorAnglesRHS(degrees):
	degreesRHS = numpy.array(degrees) 
	for i in range(degreesRHS.shape[0]):
		for j in range(degreesRHS.shape[1]):
			if isOnLeftHandSide(degreesRHS[i,j]):
				degreesRHS[i,j] = math.pi-degreesRHS[i,j]
			if degreesRHS[i,j] >= math.pi*3/2:
				degreesRHS[i,j] = degreesRHS[i,j] - math.pi*2
	return degreesRHS

# Returns a numpy.ndarray
def createHistogram(degrees, magnitudes, boundaries, bins):
	for i in range(degrees.shape[0]):
		for j in range(degrees.shape[1]):
			for b in range(bins.shape[0]):
				if(degrees[i,j] >= boundaries[b] and degrees[i,j] < boundaries[b+1]):
					bins[b] += magnitudes[i,j]
					break
	return bins

# Define bin boundaries according to number of bins.
def binBoundaries(numberOfBins):
	binValues = []
	for i in range(numberOfBins+1):
		boundary = math.radians(-90+180*i/numberOfBins)
		binValues.append(boundary)
	return binValues

bendTrainingpath =  "dataset/training/bend"
jackTrainingpath = "dataset/training/jack"
jumpTrainingpath = "dataset/training/jump"
pjumpTrainingpath = "dataset/training/pjump"
runTrainingpath = "dataset/training/run"
sideTrainingpath = "dataset/training/side"
skipTrainingpath = "dataset/training/skip"
walkTrainingpath = "dataset/training/walk"
wave1Trainingpath = "dataset/training/wave1"
wave2Trainingpath = "dataset/training/wave2"

bendTrainingpaths = os.listdir(bendTrainingpath)
jackTrainingpaths = os.listdir(jackTrainingpath)
jumpTrainingpaths = os.listdir(jumpTrainingpath)
pjumpTrainingpaths = os.listdir(pjumpTrainingpath)
runTrainingpaths = os.listdir(runTrainingpath)
sideTrainingpaths = os.listdir(sideTrainingpath)
skipTrainingpaths = os.listdir(skipTrainingpath)
walkTrainingpaths = os.listdir(walkTrainingpath)
wave1Trainingpaths = os.listdir(wave1Trainingpath)
wave2Trainingpaths = os.listdir(wave2Trainingpath)

trainingPaths = []

trainingPaths.append(bendTrainingpaths)
trainingPaths.append(jackTrainingpaths)
trainingPaths.append(jumpTrainingpaths)
trainingPaths.append(pjumpTrainingpaths)
trainingPaths.append(runTrainingpaths)
trainingPaths.append(sideTrainingpaths)
trainingPaths.append(skipTrainingpaths)
trainingPaths.append(walkTrainingpaths)
trainingPaths.append(wave1Trainingpaths)
trainingPaths.append(wave2Trainingpaths)

for path in trainingPaths:
	path.remove(".DS_Store")

# Main
numberOfBins = 32
boundaries = binBoundaries(numberOfBins)
bins = numpy.zeros(numberOfBins)
tempMeanHoofs = []

for path in runTrainingpaths:
	cap = cv.VideoCapture(runTrainingpath+"/"+path)
	print(runTrainingpath+"/"+path)
	ret, frame1 = cap.read()
	prvsImage = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)

	hoof = []
	while(True):
		ret, frame2 = cap.read()
		if not(ret):
			break

		nextImage = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
		flow = cv.calcOpticalFlowFarneback(prvsImage, nextImage, None, 0.5, 3, 7, 3, 5, 1.2, 0)
		mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])	
		
		# Create frame histogram
		angRHS = mirrorAnglesRHS(ang)
		frameHist = createHistogram(angRHS, mag, boundaries, bins)
		
		# Normalize the histogram to sum up to 1.
		frameHist = frameHist/sum(bins)
		hoof.append(frameHist)

		prvsImage = nextImage
	cap.release()

	hoof = numpy.array(hoof)
	
	tempMeanHOOF = numpy.mean(hoof, axis=0)