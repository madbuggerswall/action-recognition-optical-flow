import os
import math

import numpy
import matplotlib.pyplot as mpl

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

numberOfBins = 32
boundaries = binBoundaries(numberOfBins)

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

# for i in range(len(bendHOOFs)):
# 	hist = mpl.bar(boundaries[:numberOfBins], pjumpHOOFs[i], align="edge", width=0.05)
# 	mpl.show(hist)

# DONE: Apply PCA to the datasets seperately
# Seperately applied PCA did not good for clusters.
# DONE: Apply PCA to one dataset standart scale seperately.
# Seperately standard scaled HOOFs did not good for clusters.
# TODO Apply PCA to the datasets seperatl

allHOOFs = numpy.concatenate((bendHOOFs, jackHOOFs, jumpHOOFs, pjumpHOOFs, runHOOFs, sideHOOFs, skipHOOFs, walkHOOFs, wave1HOOFs, wave2HOOFs))

allHOOFStdScaled = StandardScaler().fit_transform(allHOOFs)
print(allHOOFs.shape)
pca = PCA(.90)
allPCs = pca.fit_transform(allHOOFStdScaled)
print(sum(pca.explained_variance_ratio_))
# print(allPCs[:,0])
pcScatter = mpl.scatter(allPCs[:,0], allPCs[:,1])
mpl.show(pcScatter)