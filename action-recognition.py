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
# 	hist = mpl.bar(boundaries[:numberOfBins], bendHOOFs[i], align="edge", width=0.05)
# 	mpl.show(hist)

pca = PCA(n_components=2)

bendPC = pca.fit_transform(bendHOOFs)
jackPC = pca.fit_transform(jackHOOFs)
jumpPC = pca.fit_transform(jumpHOOFs)
pjumpPC = pca.fit_transform(pjumpHOOFs)
runPC = pca.fit_transform(runHOOFs)
sidePC = pca.fit_transform(sideHOOFs)
skipPC = pca.fit_transform(skipHOOFs)
walkPC = pca.fit_transform(walkHOOFs)
wave1PC = pca.fit_transform(wave1HOOFs)
wave2PC = pca.fit_transform(wave2HOOFs)
