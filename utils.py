import math, numpy as np
from tqdm import tqdm
from coveredTree import CoveredGraph, randomBool

def binom(n, k):
    return

def getAvgRegret(numExperimentsToAvgOver,experimentFunction, numTotalSamples,cGraphDegree,cGraphNumLayers,cGraphInitQVals,cGraphMu,cGraphEps):
    regretList = np.zeros(numExperimentsToAvgOver)
    for i in tqdm(range(numExperimentsToAvgOver)):
        # print("iterationNumber = ",i)
        cgraph = CoveredGraph(degree=cGraphDegree, numLayers=cGraphNumLayers, initialQValues=cGraphInitQVals, mu=cGraphMu, epsilon=cGraphEps)
        regret = experimentFunction(cgraph, numTotalSamples)
        regretList[i] = regret
    regretMean = regretList.mean()
    return regretMean,regretList
