import cv2 as cv
import numpy as numpy
import matplotlib.pyplot as mpl

import math
import os

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

numberOfBins = 32
boundaries = binBoundaries(numberOfBins)
bins = numpy.zeros(numberOfBins)

cap = cv.VideoCapture("dataset/run/daria_run.avi")
ret, frame1 = cap.read()
prvsImage = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)

hsv = numpy.zeros_like(frame1)
hsv[...,1] = 255
i=0
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

	hist = mpl.bar(boundaries[:numberOfBins], frameHist, align="edge", width=0.05)
	mpl.show(hist)

	# # Visualization/HSV
	# hsv[...,0] = ang*180/math.pi/2
	# hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
	# bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
	# cv.imshow('frame2',bgr)
	# cv.waitKey(30)

	# # Visualization/Quiver
	# posX =numpy.arange(0, flow.shape[1], 2)
	# posY =numpy.arange(flow.shape[0], 0, -2)
	# quiv = mpl.quiver(posX, posY, flow[::2,::2,0], flow[::2,::2,1], scale=3e2)
	# mpl.show(quiv)
	
	# # Save quiver plots to file
	# mpl.savefig("test/test"+str(i)+".png", format="png", dpi=200)
	# mpl.clf()
	# i+=1

	prvsImage = nextImage
cap.release()
cv.destroyAllWindows()

