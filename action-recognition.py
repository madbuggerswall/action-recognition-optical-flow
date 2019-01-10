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

# TODO: Apply PCA to the datasets seperately
allHOOFs = numpy.concatenate((bendHOOFs, jackHOOFs, jumpHOOFs, pjumpHOOFs, runHOOFs, sideHOOFs, skipHOOFs, walkHOOFs, wave1HOOFs, wave2HOOFs))
allPCsSTD = StandardScaler().fit_transform(allHOOFs)

pca = PCA(n_components=2)
pca = pca.fit(allPCsSTD)
print(pca.explained_variance_ratio_)
allPCsSTD = pca.fit_transform(allPCsSTD)

# print(allPCs[:,0])
pcScatter = mpl.scatter(allPCsSTD[:,0], allPCsSTD[:,1])
mpl.show(pcScatter)