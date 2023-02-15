import numpy as np, random as rd, statistics as stats
from coveredTree import CoveredGraph, randomBool
from tqdm import tqdm
import time
from utils import getAvgRegret


def getYabeRegret(cgraph, numTotalSamples):
    numInterventionSets = cgraph.numPenultimate
    numInterventionsPerSet = 2 ** cgraph.degree
    numTimesSeen = np.zeros((numInterventionSets, numInterventionsPerSet),dtype=int)
    numTimesOne = np.zeros((numInterventionSets, numInterventionsPerSet),dtype=int)
    probOne = np.zeros((numInterventionSets, numInterventionsPerSet),dtype=float)
    avgRewardOnIntervention = np.zeros((numInterventionSets, numInterventionsPerSet), dtype=float)


    numInterventions = numInterventionSets * (2 ** cgraph.degree)
    # Number of interventions sets is equal to the number of penultimate nodes as for each
    numSamplesPerIntervention = numTotalSamples // numInterventions

    for row in range(numInterventionSets):
        for col in range(numInterventionsPerSet):
            currTimesSeen, currTimesOne = cgraph.getProbsWithDoOperation(row,col,numSamplesPerIntervention)
            numTimesSeen = numTimesSeen + currTimesSeen
            numTimesOne = numTimesOne + currTimesOne
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
    regret = 0 if np.argmax(avgRewardOnIntervention) == avgRewardOnIntervention.size-1 else cgraph.epsilon

    return regret


if __name__ == "__main__":
    startTime = time.time()
    np.random.seed(8)
    np.set_printoptions(precision=3)
    rd.seed(8)
    # cgraph = CoveredGraph.__new__(CoveredGraph)
    # cgraph.__init__(degree=3, numLayers=4, initialQValues=0.0,mu=0.05,epsilon=0.05)
    degree, numLayers, initialQValues, mu, epsilon = 3, 3, 0, 0.1, 0.05
    numTotalSamples = 20000
    numExperimentsToAvgOver = 50
    regretMean, regretList = getAvgRegret(numExperimentsToAvgOver, getYabeRegret,
                                          numTotalSamples, degree, numLayers,
                                          initialQValues, mu, epsilon)

    print("regretList=", regretList)
    print("regret=", regretMean)
    # regretList = np.zeros(numExperimentsToAvgOver)
    # for i in tqdm(range(numExperimentsToAvgOver)):
    #     # print("iterationNumber = ",i)
    #     cgraph = CoveredGraph(degree=degree, numLayers=numLayers, initialQValues=initialQValues, mu=mu, epsilon=epsilon)
    #     regret = getYabeRegret(cgraph, numTotalSamples)
    #     regretList[i] = regret
    # print("regretList=", regretList)
    # print("regret=", regretList.mean())

    print("time taken in seconds:", time.time() - startTime)
