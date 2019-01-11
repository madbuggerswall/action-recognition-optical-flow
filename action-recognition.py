import os
import math
import sys
from enum import IntEnum

import numpy
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# WARNING: This script should be in the same folder with /output-histograms.
# /output-histograms consists of precalculated histograms of optical flow.
# All HOOFs took 142 minutes to be calculated.

# TODO/DONE: Apply PCA to the datasets seperately (n_components=2)
# Seperately applied PCA was not good for clusters.
# TODO/DONE: Apply PCA to one dataset standart scale seperately (n_components=2).
# Seperately standard scaled HOOFs was not good for clusters.
# TODO/DONE: Apply PCA to the datasets seperatly preserving the variance at 90%.
# Resulting PC lists has different dimensions hence the concatenation fails.
# TODO/DONE: Apply Test STD Scaling PCA with whole dataset.
# K-NN accuracy increased to 20% from 0% :)

# For K-NN
class Label(IntEnum):
	BEND = 0
	JACK = 1
	JUMP = 2
	PJUMP = 3
	RUN = 4
	SIDE = 5
	SKIP = 6
	WALK = 7
	WAVE1 = 8
	WAVE2 = 9

# Returns a list consisting of bin edges.
def binBoundaries(numberOfBins):
	binValues = []
	for i in range(numberOfBins+1):
		boundary = math.radians(-90+180*i/numberOfBins)
		binValues.append(boundary)
	return binValues

# Returns a list consisting of every file path in given directory.
def initPathList(path):
	pathList = []
	for fileName in os.listdir(path):
		if(fileName == ".DS_Store"):
			continue
		pathList.append(os.path.join(path,fileName))
	return pathList

# Load all HOOFs inside a directory into a list.
def loadHOOF(pathList):
	dirHOOFs = []
	for filePath in pathList:
		dirHOOFs.append(numpy.load(filePath))
	return dirHOOFs

# Returns a list of numeric labels for K-NN
def initLabelList():
	labels = []
	for _ in bendHOOFs:
		labels.append(int(Label.BEND))
	for _ in jackHOOFs:
		labels.append(int(Label.JACK))
	for _ in jumpHOOFs:
		labels.append(int(Label.JUMP))
	for _ in pjumpHOOFs:
		labels.append(int(Label.PJUMP))
	for _ in runHOOFs:
		labels.append(int(Label.RUN))
	for _ in sideHOOFs:
		labels.append(int(Label.SIDE))
	for _ in skipHOOFs:
		labels.append(int(Label.SKIP))
	for _ in walkHOOFs:
		labels.append(int(Label.WALK))
	for _ in wave1HOOFs:
		labels.append(int(Label.WAVE1))
	for _ in wave2HOOFs:
		labels.append(int(Label.WAVE2))
	return labels

# Visualizing histogram sets. For inspection purposes.
def plotHistograms(hoofs):
	numberOfBins = 32
	boundaries = binBoundaries(numberOfBins)
	for i in range(len(hoofs)):
		hist = mpl.bar(boundaries[:numberOfBins], hoofs[i], align="edge", width=0.05)
		mpl.show(hist)


#	Main 

# Paths of training histogram directories
bendPath = "output-histograms/bend"
jackPath = "output-histograms/jack"
jumpPath = "output-histograms/jump"
pjumpPath = "output-histograms/pjump"
runPath = "output-histograms/run"
sidePath = "output-histograms/side"
skipPath = "output-histograms/skip"
walkPath = "output-histograms/walk"
wave1Path = "output-histograms/wave1"
wave2Path = "output-histograms/wave2"

# Path of test histogram directories
testPath = "output-histograms/test"

# Paths of training histograms as lists
bendHistPaths = initPathList(bendPath)
jackHistPaths = initPathList(jackPath)
jumpHistPaths = initPathList(jumpPath)
pjumpHistPaths = initPathList(pjumpPath)
runHistPaths = initPathList(runPath)
sideHistPaths = initPathList(sidePath)
skipHistPaths = initPathList(skipPath)
walkHistPaths = initPathList(walkPath)
wave1HistPaths = initPathList(wave1Path)
wave2HistPaths = initPathList(wave2Path)

# Paths of test histograms as a list
testHistPaths = initPathList(testPath)

# Histograms of training videos
bendHOOFs = loadHOOF(bendHistPaths)
jackHOOFs = loadHOOF(jackHistPaths)
jumpHOOFs = loadHOOF(jumpHistPaths)
pjumpHOOFs = loadHOOF(pjumpHistPaths)
runHOOFs = loadHOOF(runHistPaths)
sideHOOFs = loadHOOF(sideHistPaths)
skipHOOFs = loadHOOF(skipHistPaths)
walkHOOFs = loadHOOF(walkHistPaths)
wave1HOOFs = loadHOOF(wave1HistPaths)
wave2HOOFs = loadHOOF(wave2HistPaths)

# Histograms of test videos
testHOOFs = loadHOOF(testHistPaths)

# Plot histograms for inspection purposes.
# plotHistograms(runHOOFs)

# Creates a list of labels for training set.
labels = initLabelList()

allHOOFs = numpy.concatenate((bendHOOFs, jackHOOFs, jumpHOOFs, pjumpHOOFs, 
runHOOFs, sideHOOFs, skipHOOFs, walkHOOFs, wave1HOOFs, wave2HOOFs, testHOOFs))
print("allHOOfs shape:",allHOOFs.shape)

# Scale all histograms for PCA.
allHOOFStdScaled = StandardScaler().fit_transform(allHOOFs)

pca = PCA(.90)
allPCs = pca.fit_transform(allHOOFStdScaled)

print("allPCs shape:", allPCs.shape)
print("Component variances:", pca.explained_variance_ratio_)
print("Total preserved variance:", sum(pca.explained_variance_ratio_))

# Split all PCs into training set and test set.
trainingPCs = allPCs[:-len(testHOOFs)]
testPCs = allPCs[-len(testHOOFs):]

# Train the model.
knnModel = KNeighborsClassifier(n_neighbors=5)
knnModel.fit(trainingPCs, labels)

# Predict for all test histograms.
for i in range(len(testPCs)):
	predicted = knnModel.predict([testPCs[i]])
	print("Test data:", str(Label(i))[6:], "- Prediction:", str(Label(predicted))[6:], "\t", predicted==i)

# Visualize the 5D data on 2D plane by the first 2 columns. 
fig = mpl.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(trainingPCs[:,0], trainingPCs[:,1], s=10, c='b', marker="s", label='Training')
ax1.scatter(testPCs[:,0],testPCs[:,1], s=10, c='r', marker="o", label='Test')
mpl.legend(loc='upper left')
mpl.show()
