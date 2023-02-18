import numpy as np, random as rd, statistics as stats
from coveredTree import CoveredGraph, randomBool
from tqdm import tqdm
import time
from utils import getAvgRegret

def getCIRegret(cgraph, numTotalSamples):
    numTimesSeen, numTimesOne = cgraph.getProbsWithDoOperationUsingCovers(numTotalSamples)

    numInterventionSets = cgraph.numPenultimate
    numInterventionsPerSet = 2 ** cgraph.degree
    probOne = np.zeros((numInterventionSets, numInterventionsPerSet), dtype=float)
    avgRewardOnIntervention = np.zeros((numInterventionSets, numInterventionsPerSet), dtype=float)
    for row in range(numInterventionSets):
        for col in range(numInterventionsPerSet):
            probOne[row,col] = numTimesOne[row,col]/numTimesSeen[row,col]
    # print("numTimesSeen=",numTimesSeen)
    # print("numTimesOne=", numTimesOne)
    # print("probOne=", probOne)

    for row in range(numInterventionSets):
        for col in range(numInterventionsPerSet):
            prod = 1
            for penultimateIndex in range(cgraph.numPenultimate):
                if(penultimateIndex==row):
                    prod = prod * (1-probOne[penultimateIndex, col])
                else:
                    prod = prod* (1-probOne[penultimateIndex,0])

            avgRewardOnIntervention[row,col] = 1 - prod

    # print("avgRewardOnIntervention=", avgRewardOnIntervention)
    # print("np.argmax(avgRewardOnIntervention)=", np.argmax(avgRewardOnIntervention))
    # print("avgRewardOnIntervention.size=", avgRewardOnIntervention.size)
    # regret = 0 if np.argmax(avgRewardOnIntervention) == avgRewardOnIntervention.size-1 else cgraph.epsilon
    regret = 0 if np.argmax(avgRewardOnIntervention) == avgRewardOnIntervention.size-1 else cgraph.regretOnChoosingBadIntervention

    return regret


if __name__ == "__main__":
    startTime = time.time()
    np.random.seed(8)
    np.set_printoptions(precision=3)
    rd.seed(8)
    # cgraph = CoveredGraph.__new__(CoveredGraph)
    # cgraph.__init__(degree=3, numLayers=4, initialQValues=0.0,pi=0.05,epsilon=0.05)
    degree, numLayers, initialQValues, pi, epsilon = 3, 3, 0, 0.1, 0.2
    numTotalSamples = 200
    numExperimentsToAvgOver = 50
    regretMean, regretList = getAvgRegret(numExperimentsToAvgOver, getCIRegret,
                                          numTotalSamples, degree, numLayers,
                                          initialQValues, pi, epsilon)

    print("regretList=", regretList)
    print("regret=", regretMean)

    # regretList = np.zeros(numExperimentsToAvgOver)
    # for i in tqdm(range(numExperimentsToAvgOver)):
    #     # print("iterationNumber = ",i)
    #     cgraph = CoveredGraph(degree=degree, numLayers=numLayers, initialQValues=initialQValues, pi=pi, epsilon=epsilon)
    #     regret = getCIRegret(cgraph, numTotalSamples)
    #     regretList[i] = regret
    # print("regretList=", regretList)
    # print("regret=", regretList.mean())

    print("time taken in seconds:", time.time() - startTime)
