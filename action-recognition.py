import os
import math
from enum import IntEnum

import numpy
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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

def binBoundaries(numberOfBins):
	binValues = []
	for i in range(numberOfBins+1):
		boundary = math.radians(-90+180*i/numberOfBins)
		binValues.append(boundary)
	return binValues

def initPathList(path):
	pathList = []
	for fileName in os.listdir(path):
		if(fileName == ".DS_Store"):
			continue
		pathList.append(os.path.join(path,fileName))
	return pathList

def loadHOOF(pathList):
	dirHOOFs = []
	for filePath in pathList:
		dirHOOFs.append(numpy.load(filePath))
	return dirHOOFs

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


#	Main 

numberOfBins = 32
boundaries = binBoundaries(numberOfBins)

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


labels = initLabelList()
print(labels)

# for i in range(len(bendHOOFs)):
# 	hist = mpl.bar(boundaries[:numberOfBins], pjumpHOOFs[i], align="edge", width=0.05)
# 	mpl.show(hist)

# DONE: Apply PCA to the datasets seperately (n_components=2)
# Seperately applied PCA did not good for clusters.
# DONE: Apply PCA to one dataset standart scale seperately (n_components=2).
# Seperately standard scaled HOOFs did not good for clusters.
# DONE: Apply PCA to the datasets seperatly preserving the variance at 90%.
# Resulting PC lists has different dimensions hence the concatenation fails.
# UNKNOWN: Representing a dataset with smaller datasets with different dimensions.

allHOOFs = numpy.concatenate((bendHOOFs, jackHOOFs, jumpHOOFs, pjumpHOOFs, 
runHOOFs, sideHOOFs, skipHOOFs, walkHOOFs, wave1HOOFs, wave2HOOFs))
print("allHOOfs shape:",allHOOFs.shape)


allHOOFStdScaled = StandardScaler().fit_transform(allHOOFs)
testHOOFStdScaled = StandardScaler().fit_transform(testHOOFs)

pcaTraining = PCA(.90)
allPCs = pcaTraining.fit_transform(allHOOFStdScaled)

print("Training / allPCs shape:", allPCs.shape)
print("Training / Component variances:", pcaTraining.explained_variance_ratio_)
print("Training / Total preserved variance:", sum(pcaTraining.explained_variance_ratio_))

# Minimum of 90% reserved variance resulted as 3 components.
# Manually take 5 components to avoid dimension problems.

pcaTest = PCA(n_components=5)
testPCs = pcaTest.fit_transform(testHOOFStdScaled)

print("Test / testPCs shape:", testPCs.shape)
print("Test / Component variances:", pcaTest.explained_variance_ratio_)
print("Test / Total preserved variance:", sum(pcaTest.explained_variance_ratio_))

knnModel = KNeighborsClassifier(n_neighbors=3)
knnModel.fit(allPCs, labels)

predicted = knnModel.predict(testPCs[0].reshape(1,-1))
print(predicted)
predicted = knnModel.predict(testPCs[1].reshape(1,-1))
print(predicted)
predicted = knnModel.predict(testPCs[2].reshape(1,-1))
print(predicted)
predicted = knnModel.predict(testPCs[3].reshape(1,-1))
print(predicted)
predicted = knnModel.predict(testPCs[4].reshape(1,-1))
print(predicted)
predicted = knnModel.predict(testPCs[5].reshape(1,-1))
print(predicted)
predicted = knnModel.predict(testPCs[6].reshape(1,-1))
print(predicted)
predicted = knnModel.predict(testPCs[7].reshape(1,-1))
print(predicted)
predicted = knnModel.predict(testPCs[8].reshape(1,-1))
print(predicted)
predicted = knnModel.predict(testPCs[9].reshape(1,-1))
print(predicted)


# Visualize the 5D data on 2D plane by the first 2 columns. 
fig = mpl.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(allPCs[:,0], allPCs[:,1], s=10, c='b', marker="s", label='first')
ax1.scatter(testPCs[:,0],testPCs[:,1], s=10, c='r', marker="o", label='second')
mpl.legend(loc='upper left')
mpl.show()
