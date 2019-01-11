import math
import os

import cv2 as cv
import numpy 
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

# Returns a list of the first videos in the folders. (Alphabetically)
def getVideoPaths():
	videoPaths = []
	datasetPath = "dataset/training"
	dirNames = os.listdir(datasetPath)
	dirNames.sort()
	for dirName in dirNames:
		dirPath = os.path.join(datasetPath, dirName)
		if os.path.isdir(dirPath):
			fileNames = os.listdir(dirPath)
			fileNames.sort()
			for fileName in fileNames:
				if fileName == ".DS_Store":
					continue
				videoPaths.append(os.path.join(dirPath,fileName))
				break
	return videoPaths

# Main

numberOfBins = 32
boundaries = binBoundaries(numberOfBins)

videoPaths = getVideoPaths()
outputRoot = "output-visuals"

for videoPath in videoPaths:
	# Create visual output folder
	outputDir = os.path.basename(os.path.dirname(videoPath))
	outputDirPath = os.path.join(outputRoot, outputDir)
	if not(os.path.exists(outputDirPath)):
		os.makedirs(outputDirPath)

	videoName = os.path.splitext(os.path.basename(videoPath))[0]
	outputPath = os.path.join(outputDirPath, videoName)

	# First frame for optical flow.
	cap = cv.VideoCapture(videoPath)
	ret, frame1 = cap.read()
	prvsImage = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)

	# For HSV visualization.
	hsv = numpy.zeros_like(frame1)
	hsv[...,1] = 255
	
	# Get the first five frames of the video.
	for i in range(5):
		bins = numpy.zeros(numberOfBins)

		ret, frame2 = cap.read()
		# Write to file
		cv.imwrite(outputPath+str(i)+".png", frame2)
		if not(ret):
			break

		#	Optical flow.
		nextImage = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
		flow = cv.calcOpticalFlowFarneback(prvsImage, nextImage, None, 0.5, 3, 7, 3, 5, 1.2, 0)
		mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])	
		
		# Create frame histogram
		angRHS = mirrorAnglesRHS(ang)
		frameHist = createHistogram(angRHS, mag, boundaries, bins)
		
		# Normalize the histogram to sum up to 1.
		frameHist = frameHist/sum(bins)

		# Visualization/Histogram & File output
		hist = mpl.bar(boundaries[:numberOfBins], frameHist, align="edge", width=0.05)
		# Write to file
		mpl.savefig(outputPath+"HOOF"+str(i)+".png", format="png", dpi=200)
		mpl.clf()

		# Visualization/HSV & File output
		hsv[...,0] = ang*180/math.pi/2
		hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
		opFlowHSV = cv.cvtColor(hsv,cv.COLOR_HSV2RGB)
		# Write to file
		cv.imwrite(outputPath+"HSV"+str(i)+".png", opFlowHSV)

		# Visualization/Quiver & File output
		posX =numpy.arange(0, flow.shape[1], 2)
		posY =numpy.arange(flow.shape[0], 0, -2)
		quiv = mpl.quiver(posX, posY, flow[::2,::2,0], flow[::2,::2,1], scale=3e2)
		mpl.savefig(outputPath+"QUIV"+str(i)+".png", format="png", dpi=200)
		mpl.clf()
		
		prvsImage = nextImage
	cap.release()