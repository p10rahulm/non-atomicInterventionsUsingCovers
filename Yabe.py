import numpy as np, random as rd, statistics as stats
from coveredTree import CoveredGraph, randomBool
from tqdm import tqdm

def getYabeRegret(cgraph, numTotalSamples):
    numInterventionSets = cgraph.numPenultimate
    numInterventions = numInterventionSets * (2 ** cgraph.degree)
    # Number of interventions sets is equal to the number of penultimate nodes as for each
    numSamplesPerIntervention = numTotalSamples // numInterventions

    # print("cgraph=", cgraph)
    # print("numSamplesPerIntervention=", numSamplesPerIntervention)
    avgRewardsList = np.zeros(numInterventions)
    listIndex = 0
    for penultimateIndex in cgraph.penultimateLayerSet:
        children = cgraph.getChildIndices(penultimateIndex)
        assignments = cgraph.assignmentSet
        for assignment in assignments:
            rewardsOnDo = []
            for sampleNum in range(numSamplesPerIntervention):
                graphObs = cgraph.doOperation(children, assignment)
                rewardsOnDo.append(graphObs[0])
            # print("penultimateIndex=", penultimateIndex, "children=", children, "assignment=", assignment,
            #       "avg reward=", stats.fmean(rewardsOnDo))
            avgReward = stats.fmean(rewardsOnDo)
            avgRewardsList[listIndex] = avgReward
            listIndex += 1
    # print("avgRewardsList=",avgRewardsList)
    maxIndex = np.argmax(avgRewardsList)
    # print("maxIndex=", maxIndex)
    (penultimateIndex, childrenOfPenultimate, assignmentIndex, assignment) = cgraph.getInterventionFromIndex(maxIndex)
    # print("penultimateIndex=", penultimateIndex, "children=", childrenOfPenultimate, "assignment=", assignment,
    #       "assignmentIndex=", assignmentIndex, "avg reward=", avgRewardsList[maxIndex])

    regret = 0 if (penultimateIndex == cgraph.penultimateLayerEndIndex and assignmentIndex == 7) else cgraph.epsilon
    return regret


if __name__ == "__main__":
    np.random.seed(8)
    np.set_printoptions(precision=3)
    rd.seed(8)
    # cgraph = CoveredGraph.__new__(CoveredGraph)
    # cgraph.__init__(degree=3, numLayers=4, initialQValues=0.0,mu=0.05,epsilon=0.05)
    degree, numLayers, initialQValues, mu, epsilon = 3, 3, 0, 0.05, 0.05
    numTotalSamples = 100000
    numExperimentsToAvgOver = 50
    regretList = np.zeros(numExperimentsToAvgOver)
    for i in tqdm(range(numExperimentsToAvgOver)):
        # print("iterationNumber = ",i)
        cgraph = CoveredGraph(degree=degree, numLayers=numLayers, initialQValues=initialQValues, mu=mu, epsilon=epsilon)
        regret = getRegret(cgraph, numTotalSamples)
        regretList[i] = regret
    print("regretList=", regretList)
    print("regret=", regretList.mean())
