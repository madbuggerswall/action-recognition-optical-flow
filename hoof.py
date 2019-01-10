import math
import sys
import os

import cv2 as cv
import numpy as numpy
import matplotlib.pyplot as mpl

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

# Returns a numpy.ndarray (HOOF)
def createHistogram(degrees, magnitudes, boundaries, bins):
	for i in range(degrees.shape[0]):
		for j in range(degrees.shape[1]):
			for b in range(bins.shape[0]):
				if(degrees[i,j] >= boundaries[b] and degrees[i,j] < boundaries[b+1]):
					bins[b] += magnitudes[i,j]
					break
	return bins

# Returns a list consisting of bin edges.
def binBoundaries(numberOfBins):
	binValues = []
	for i in range(numberOfBins+1):
		boundary = math.radians(-90+180*i/numberOfBins)
		binValues.append(boundary)
	return binValues

# Returns the list of videos to be analyzed.
def initVideoPathList(directoryPath):
	videoPaths = []
	for dirName in os.listdir(directoryPath):
		if(dirName == ".DS_Store"):
			continue
		if os.path.isfile(os.path.join(directoryPath, dirName)):
			videoPaths.append(dirName)
			continue
		for fileName in os.listdir(os.path.join(directoryPath, dirName)):
			if(fileName == ".DS_Store"):
				continue
			videoPaths.append(os.path.join(dirName, fileName))
	return videoPaths


# Main

# If no arguments given go through training set.
if len(sys.argv) == 1:
	directoryPath = "dataset/training/run"
	fileSavePath = "output-histograms"

# Test set and desired histogram output directory for test set.
if len(sys.argv) == 3:
	directoryPath = sys.argv[1]
	fileSavePath = sys.argv[2]

# Number of histogram bins to classify the angles.
numberOfBins = 32
boundaries = binBoundaries(numberOfBins)
videoPaths = initVideoPathList(directoryPath)

for videoPath in videoPaths:
	print(os.path.join(directoryPath, videoPath))
	
	cap = cv.VideoCapture(os.path.join(directoryPath, videoPath))
	ret, frame1 = cap.read()
	prvsImage = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)

	# Empty the list of histograms once a video.
	hoofAllFrames = []
	
	while(True):
		bins = numpy.zeros(numberOfBins)
		ret, frame2 = cap.read()
		if not(ret):
			break

		# Calculate optical flow.
		nextImage = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
		flow = cv.calcOpticalFlowFarneback(prvsImage, nextImage, None, 0.5, 3, 7, 3, 5, 1.2, 0)
		mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])	
		
		# Create frame histogram
		angRHS = mirrorAnglesRHS(ang)
		frameHist = createHistogram(angRHS, mag, boundaries, bins)
		
		# Normalize the histogram to sum up to 1.
		frameHist = frameHist/sum(frameHist)
		hoofAllFrames.append(frameHist)

		prvsImage = nextImage
	cap.release()

	hoofAllFrames = numpy.array(hoofAllFrames)
	tempMeanHOOF = numpy.mean(hoofAllFrames, axis=0)

	# Saves the histogram in a file for later use.
	savePath = os.path.join(fileSavePath, videoPath[:-4])
	numpy.save(savePath+".npy", tempMeanHOOF)