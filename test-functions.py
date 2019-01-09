import cv2
import numpy
import os

class Degree:
	def __init__(self, value):
		self.value = value % 360.0
		if self.value < 0:
			self.value += 360.0

	def isInInterval(self, lowerBound, upperBound):
		degree = Degree(self.value)
		if upperBound < lowerBound:
			if degree < upperBound:
				degree += 360
			upperBound += 360
		return degree >= lowerBound and degree < upperBound

	#	Overload (+) operator
	def __add__(self, other):
		if other is Degree:
			return Degree(self.value + other.value)
		else: 
			return Degree(self.value + other)

	#	Overload (-) operator
	def __sub__(self, other):
		if other is Degree:
			return Degree(self.value - other.value)
		else:
			return Degree(self.value - other)

	# Overload (*) operator
	def __mul__(self, other):
		if other is Degree:
			return Degree(self.value * other.value)
		else:
			return Degree(self.value * other)

	# Overload (/) operator
	def __truediv__(self, other):
		if other is Degree:
			return Degree(self.value / other.value)
		else:
			return Degree(self.value / other)

	#	Little than	
	def __lt__(self, other):
		if other is Degree:
			return self.value < other.value
		else:
			return self.value < other

	#	Little than or equal
	def __le__(self, other):
		if other is Degree:
			return self.value <= other.value
		else:
			return self.value <= other

	#	Equal
	def __eq__(self, other):
		if other is Degree:
			return self.value == other.value
		else:
			return self.value == other

		
	#	Not equal
	def __ne__(self, other):
		if other is Degree:
			return self.value != other.value
		else:
			return self.value != other
		
	#	Greater than
	def	__gt__(self, other):
		if other is Degree:
			return self.value > other.value
		else:
			return self.value > other
		
	#	Greater than or equal
	def __ge__(self, other):
		if other is Degree:
			return self.value >= other.value
		else:
			return self.value >= other

	#	String overload for print(Degree)
	def __str__(self):
		return str(self.value)

def dispOpticalFlow( Image,Flow,Divisor,name ):
	PictureShape = numpy.shape(Image)
	#determine number of quiver points there will be
	Imax = int(PictureShape[0]/Divisor)
	Jmax = int(PictureShape[1]/Divisor)
	#create a blank mask, on which lines will be drawn.
	mask = numpy.zeros_like(Image)
	for i in range(1, Imax):
		for j in range(1, Jmax):
			X1 = (i)*Divisor
			Y1 = (j)*Divisor
			X2 = int(X1 + Flow[X1,Y1,1])
			Y2 = int(Y1 + Flow[X1,Y1,0])
			X2 = numpy.clip(X2, 0, PictureShape[0])
			Y2 = numpy.clip(Y2, 0, PictureShape[1])
			#add all the lines to the mask
			mask = cv2.line(mask, (Y1,X1),(Y2,X2), [255, 255, 255], 1)
	#superpose lines onto image
	img = cv2.add(Image,mask)
	#print image
	cv2.imshow(name,img)
	return []

trainingDirs = []
trainingPath = "dataset/training"
for dirName in os.listdir(trainingPath):
	if(dirName == ".DS_Store"):
		continue
	for fileName in os.listdir(os.path.join(trainingPath, dirName)):
		if(fileName == ".DS_Store"):
			continue
		trainingDirs.append(os.path.join(dirName, fileName))
		print(os.path.join(dirName, fileName))