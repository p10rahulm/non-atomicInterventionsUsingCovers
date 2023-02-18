# from pureBandit import getPureBanditRegret, getPureBanditRegret2
# from Yabe import getYabeRegret
# from CoveringInterventions import getCIRegret
# from utils import getAvgRegret
import time, numpy as np, random as rd, statistics as stats, pandas as pd
from coveredTree import CoveredGraph, randomBool
from tqdm import tqdm
import math, collections


#
# def getCIRegret(cgraph, numTotalSamples):
#     numTimesSeen, numTimesOne = cgraph.getProbsWithDoOperationUsingCovers(numTotalSamples)
#
#     numInterventionSets = cgraph.numPenultimate
#     numInterventionsPerSet = 2 ** cgraph.degree
#     probOne = np.zeros((numInterventionSets, numInterventionsPerSet), dtype=float)
#     avgRewardOnIntervention = np.zeros((numInterventionSets, numInterventionsPerSet), dtype=float)
#     for row in range(numInterventionSets):
#         for col in range(numInterventionsPerSet):
#             probOne[row, col] = numTimesOne[row, col] / numTimesSeen[row, col]
#     # print("numTimesSeen=",numTimesSeen)
#     # print("numTimesOne=", numTimesOne)
#     # print("probOne=", probOne)
#
#     for row in range(numInterventionSets):
#         for col in range(numInterventionsPerSet):
#             prod = 1
#             for penultimateIndex in range(cgraph.numPenultimate):
#                 if (penultimateIndex == row):
#                     prod = prod * (1 - probOne[penultimateIndex, col])
#                 else:
#                     prod = prod * (1 - probOne[penultimateIndex, 0])
#
#             avgRewardOnIntervention[row, col] = 1 - prod
#
#     regret = 0 if np.argmax(
#         avgRewardOnIntervention) == avgRewardOnIntervention.size - 1 else cgraph.regretOnChoosingBadIntervention
#
#     return regret


def getAvgRegret(experimentFunction, numTotalSamples, numExperimentsToAvgOver=100, numPenultimate=9,
                 degree=3, pi=0.001, epsilon=0.05, probOf1AtLeaves=0):
    regretList = np.zeros(numExperimentsToAvgOver)
    for i in tqdm(range(numExperimentsToAvgOver)):
        # print("iterationNumber = ",i)
        regret = experimentFunction(numTotalSamples=numTotalSamples, numPenultimate=numPenultimate,
                                    degree=degree, pi=pi, epsilon=epsilon,
                                    probOf1AtLeaves=probOf1AtLeaves)
        regretList[i] = regret
    regretMean = regretList.mean()
    return regretMean, regretList


def getDirectExpRegret(numTotalSamples, numPenultimate, degree=3, pi=0.001, epsilon=0.05, probOf1AtLeaves=0):
    numInterventionSets = numPenultimate
    numInterventionsPerSet = (2 ** degree)
    numInterventions = numInterventionSets * (2 ** degree)
    numSamplesPerIntervention = numTotalSamples // numInterventions

    # print("numSamplesPerIntervention=", numSamplesPerIntervention)
    probOneEstimate = np.zeros((numInterventionSets, numInterventionsPerSet), dtype=float)

    probOfOneGoodIntervention = 1 - (1 - pi - epsilon) * (1 - pi) ** (numPenultimate - 1)
    probOfOneBadIntervention = 1 - ((1 - pi) ** (numPenultimate) * (1 - probOf1AtLeaves ** 2) +
                                    (1 - pi) ** (numPenultimate - 1) * (1 - pi - epsilon) * (probOf1AtLeaves ** 2))
    regretForBadIntervention = probOfOneGoodIntervention - probOfOneBadIntervention

    # print("probOfOneGoodIntervention=", round(probOfOneGoodIntervention, 4),
    #       "probOfOneBadIntervention=", round(probOfOneBadIntervention, 4),
    #       "regretForBadIntervention=", round(regretForBadIntervention, 4))
    for row in range(numInterventionSets):
        for col in range(numInterventionsPerSet):
            numTimesSeen = numSamplesPerIntervention
            if row == numInterventionSets - 1 and col == numInterventionsPerSet - 1:
                probOf1 = probOfOneGoodIntervention
            else:
                probOf1 = probOfOneBadIntervention
            numTimesOne = np.random.binomial(n=numTimesSeen, p=probOf1)
            probOneEstimate[row, col] = numTimesOne / numTimesSeen
    # print("probOneEstimate=", probOneEstimate)
    maxIndex = np.argmax(probOneEstimate)
    maxRow = maxIndex // numInterventionsPerSet
    maxCol = maxIndex % numInterventionsPerSet
    # print("maxRow=", maxRow, "maxCol=", maxCol)

    regret = 0 if \
        (maxRow == numInterventionSets - 1 and maxCol == numInterventionsPerSet - 1) else regretForBadIntervention
    # print("regret=", regret)
    return regret

#
# def sampleLeaves(numSamples, degree, probOf1AtLeaves):
#     # print("numSamples=",numSamples)
#     probArray = []
#     for i in range(degree + 1):
#         numTimes = math.comb(degree, i)
#         value = (1 - probOf1AtLeaves) ** (degree - i) * probOf1AtLeaves ** i
#         probArray.extend([value] * numTimes)
#     # print("probarray=",probArray)
#     choiceArray = np.random.choice(2 ** degree, numSamples, p=probArray)
#     # print("choiceArray=", choiceArray)
#     countOfChoicesDict = collections.Counter(choiceArray)
#     # print("countOfChoicesDict=", countOfChoicesDict)
#     countOfChoicesOrderedArray = []
#     for i in range(2 ** degree):
#         countOfChoicesOrderedArray.append(countOfChoicesDict[i])
#     # print("countOfChoicesOrderedArray=", countOfChoicesOrderedArray)
#     # countOfChoicesOrderedArray = [countOfChoicesDict[x] for x in sorted(countOfChoicesDict.keys())]
#     return countOfChoicesOrderedArray
#
#
# def getProbsWithDoOperation(opPenultimateIndex, opPenultimateValueIndex, numTimesIntervened, numPenultimate, degree, pi,
#                             epsilon, probOf1AtLeaves):
#     # if opPenultimateIndex < 0 or opPenultimateValueIndex >= numPenultimate or \
#     #         opPenultimateValueIndex < 0 or opPenultimateValueIndex >= 2 ** degree:
#     #     raise Exception("Sorry, not a valid do() operation")
#     numTimesSeen = np.zeros((numPenultimate, 2 ** degree))
#     numTimesOne = np.zeros((numPenultimate, 2 ** degree))
#     for penultimateIndex in range(numPenultimate):
#         if penultimateIndex != opPenultimateIndex:
#             # print("opPenultimateIndex=",opPenultimateIndex,"opPenultimateValueIndex=",opPenultimateValueIndex)
#             returnedSamples = sampleLeaves(numSamples, degree, probOf1AtLeaves)
#             # print("numTimesIntervened=", numTimesIntervened,"returnedSamples=",returnedSamples)
#             numTimesSeen[penultimateIndex] = returnedSamples
#         else:
#             numTimesSeen[penultimateIndex, opPenultimateValueIndex] = numTimesIntervened
#
#     for row in range(numPenultimate):
#         for col in range(2 ** degree):
#             if row == numPenultimate - 1 and col == 2 ** degree - 1:
#                 numTimesOne[row, col] = np.random.binomial(n=numTimesSeen[row, col], p=pi + epsilon)
#             else:
#                 numTimesOne[row, col] = np.random.binomial(n=numTimesSeen[row, col], p=pi)
#
#     return numTimesSeen, numTimesOne


def getYabeRegret(numTotalSamples, numPenultimate, degree=3, pi=0.001, epsilon=0.05, probOf1AtLeaves=0):
    numInterventionSets = numPenultimate
    numInterventionsPerSet = (2 ** degree)
    numInterventions = numInterventionSets * (2 ** degree)
    numSamplesPerIntervention = numTotalSamples // numInterventions

    numTimesSeen = np.zeros((numInterventionSets, numInterventionsPerSet), dtype=int)
    numTimesOne = np.zeros((numInterventionSets, numInterventionsPerSet), dtype=int)
    # print("numSamplesPerIntervention=", numSamplesPerIntervention)
    probOneEstimate = np.zeros((numInterventionSets, numInterventionsPerSet), dtype=float)
    avgRewardEstimate = np.zeros((numInterventionSets, numInterventionsPerSet), dtype=float)
    probOfOneGoodIntervention = 1 - (1 - pi - epsilon) * (1 - pi) ** (numPenultimate - 1)
    probOfOneBadIntervention = 1 - ((1 - pi) ** (numPenultimate) * (1 - probOf1AtLeaves ** 2) +
                                    (1 - pi) ** (numPenultimate - 1) * (1 - pi - epsilon) * (probOf1AtLeaves ** 2))
    regretForBadIntervention = probOfOneGoodIntervention - probOfOneBadIntervention

    # print("probOfOneGoodIntervention=", round(probOfOneGoodIntervention, 4),
    #       "probOfOneBadIntervention=", round(probOfOneBadIntervention, 4),
    #       "regretForBadIntervention=", round(regretForBadIntervention, 4))

    for row in range(numInterventionSets):
        for col in range(numInterventionsPerSet):
            for penultimateIndex in range(numInterventionSets):
                if row==penultimateIndex:
                    numTimesSeen[penultimateIndex,col]+=numSamplesPerIntervention
                else:
                    numTimesSeen[penultimateIndex, 0] += numSamplesPerIntervention
    for row in range(numInterventionSets):
        for col in range(numInterventionsPerSet):
            if row == numPenultimate - 1 and col == numInterventionsPerSet - 1:
                numTimesOne[row, col] = np.random.binomial(n=numTimesSeen[row, col], p=pi + epsilon)
            else:
                numTimesOne[row, col] = np.random.binomial(n=numTimesSeen[row, col], p=pi)


    probOneEstimate = numTimesOne/numTimesSeen
    # for row in range(numInterventionSets):
    #     for col in range(numInterventionsPerSet):
    #         probOneEstimate[row, col] = numTimesOne[row, col] / numTimesSeen[row, col]

    for row in range(numInterventionSets):
        for col in range(numInterventionsPerSet):
            prod = 1
            for penultimateIndex in range(numInterventionSets):
                if (penultimateIndex == row):
                    prod = prod * (1 - probOneEstimate[penultimateIndex, col])
                else:
                    prod = prod * (1 - probOneEstimate[penultimateIndex, 0])

            avgRewardEstimate[row, col] = 1 - prod

    # print("avgRewardOnIntervention=", avgRewardOnIntervention)
    # print("np.argmax(avgRewardOnIntervention)=", np.argmax(avgRewardOnIntervention))
    # print("avgRewardOnIntervention.size=", avgRewardOnIntervention.size)
    # regret = 0 if np.argmax(avgRewardOnIntervention) == avgRewardOnIntervention.size-1 else cgraph.epsilon

    # print("probOneEstimate=", probOneEstimate)
    maxIndex = np.argmax(avgRewardEstimate)
    maxRow = maxIndex // numInterventionsPerSet
    maxCol = maxIndex % numInterventionsPerSet
    # print("maxRow=", maxRow, "maxCol=", maxCol)

    regret = 0 if \
        (maxRow == numInterventionSets - 1 and maxCol == numInterventionsPerSet - 1) else regretForBadIntervention
    # print("regret=", regret)
    return regret


def sampleLeavesUniformly(numSamples,numInterventionsPerSet):
    choiceArray = np.random.choice(numInterventionsPerSet, numSamples)
    countOfChoicesDict = collections.Counter(choiceArray)
    # print("countOfChoicesDict=", countOfChoicesDict)
    countOfChoicesOrderedArray = []
    for i in range(numInterventionsPerSet):
        countOfChoicesOrderedArray.append(countOfChoicesDict[i])

    return countOfChoicesOrderedArray

def getCIRegret(numTotalSamples, numPenultimate, degree=3, pi=0.001, epsilon=0.05, probOf1AtLeaves=0):
    numInterventionSets = numPenultimate
    numInterventionsPerSet = (2 ** degree)
    numInterventions = numInterventionSets * (2 ** degree)
    numSamplesPerIntervention = numTotalSamples // numInterventions

    numTimesSeen = np.zeros((numInterventionSets, numInterventionsPerSet), dtype=int)
    numTimesOne = np.zeros((numInterventionSets, numInterventionsPerSet), dtype=int)
    # print("numSamplesPerIntervention=", numSamplesPerIntervention)
    probOneEstimate = np.zeros((numInterventionSets, numInterventionsPerSet), dtype=float)
    avgRewardEstimate = np.zeros((numInterventionSets, numInterventionsPerSet), dtype=float)
    probOfOneGoodIntervention = 1 - (1 - pi - epsilon) * (1 - pi) ** (numPenultimate - 1)
    probOfOneBadIntervention = 1 - ((1 - pi) ** (numPenultimate) * (1 - probOf1AtLeaves ** 2) +
                                    (1 - pi) ** (numPenultimate - 1) * (1 - pi - epsilon) * (probOf1AtLeaves ** 2))
    regretForBadIntervention = probOfOneGoodIntervention - probOfOneBadIntervention

    # print("probOfOneGoodIntervention=", round(probOfOneGoodIntervention, 4),
    #       "probOfOneBadIntervention=", round(probOfOneBadIntervention, 4),
    #       "regretForBadIntervention=", round(regretForBadIntervention, 4))

    for penultimateIndex in range(numInterventionSets):
        numTimesSeen[penultimateIndex]=sampleLeavesUniformly(numTotalSamples,numInterventionsPerSet)

    for row in range(numInterventionSets):
        for col in range(numInterventionsPerSet):
            if row == numPenultimate - 1 and col == numInterventionsPerSet - 1:
                numTimesOne[row, col] = np.random.binomial(n=numTimesSeen[row, col], p=pi + epsilon)
            else:
                numTimesOne[row, col] = np.random.binomial(n=numTimesSeen[row, col], p=pi)


    probOneEstimate = numTimesOne/numTimesSeen
    # for row in range(numInterventionSets):
    #     for col in range(numInterventionsPerSet):
    #         probOneEstimate[row, col] = numTimesOne[row, col] / numTimesSeen[row, col]

    for row in range(numInterventionSets):
        for col in range(numInterventionsPerSet):
            prod = 1
            for penultimateIndex in range(numInterventionSets):
                if (penultimateIndex == row):
                    prod = prod * (1 - probOneEstimate[penultimateIndex, col])
                else:
                    prod = prod * (1 - probOneEstimate[penultimateIndex, 0])

            avgRewardEstimate[row, col] = 1 - prod

    # print("avgRewardOnIntervention=", avgRewardOnIntervention)
    # print("np.argmax(avgRewardOnIntervention)=", np.argmax(avgRewardOnIntervention))
    # print("avgRewardOnIntervention.size=", avgRewardOnIntervention.size)
    # regret = 0 if np.argmax(avgRewardOnIntervention) == avgRewardOnIntervention.size-1 else cgraph.epsilon

    # print("probOneEstimate=", probOneEstimate)
    maxIndex = np.argmax(avgRewardEstimate)
    maxRow = maxIndex // numInterventionsPerSet
    maxCol = maxIndex % numInterventionsPerSet
    # print("maxRow=", maxRow, "maxCol=", maxCol)

    regret = 0 if \
        (maxRow == numInterventionSets - 1 and maxCol == numInterventionsPerSet - 1) else regretForBadIntervention
    # print("regret=", regret)
    return regret



if __name__ == "__main__":
    startTime = time.time()
    np.random.seed(8)
    np.set_printoptions(precision=5)
    rd.seed(8)

    # degree, numLayers, initialQValues, pi, epsilon = 2, 3, 0, 0.125, 0.125
    # numTotalSamples = 2000
    # numExperimentsToAvgOver = 1000
    # methods = [getPureBanditRegret,getYabeRegret,getCIRegret]
    # methods = [getCIRegret]

    # methods, degree, pi, epsilon, initialQValues, numTotalSamples, listOfNumberOfLayers, numExperimentsToAvgOver = \
    #     [getPureBanditRegret], 2, 0.1, 0.2, 0, 1e6, list(range(3, 8)), 1000
    # degree, pi, epsilon, initialQValues, numExperimentsToAvgOver = 2, 0.1, 0.2, 0, 10000
    # degree, pi, epsilon, initialQValues, numExperimentsToAvgOver = 2, 0.1, 0.2, 0, 10000
    numExperimentsToAvgOver, degree, pi, epsilon, initialQValues = 1000, 3, 0.001, 0.05, 0
    numPenultimateList = list(range(10, 51, 5))
    # numPenultimateList = list(range(20, 21, 10))
    # methodsTuple = [(getDirectExpRegret, 3e4, 10000), (getYabeRegret, 5000, 1000), (getCIRegret, 100, 10000)]
    # methodsTuple = [(getDirectExpRegret, 3e4, 10000)]
    # methodsTuple = [(getYabeRegret, 5000, 1000)]
    # methodsTuple = [(getCIRegret, 250, 1000)]
    methodsTuple = [(getDirectExpRegret, 3e4, 10000), (getYabeRegret, 5000, 10000), (getCIRegret, 250, 10000)]

    regretCompiled = np.zeros((len(numPenultimateList), len(methodsTuple)))
    numNodes = set()
    for i in range(len(numPenultimateList)):
        numPenultimate = numPenultimateList[i]
        numNodes.add(numPenultimate * (degree+2))
        probOfOneGoodIntervention = 1 - (1 - pi - epsilon) * (1 - pi) ** (numPenultimate - 1)
        probOfOneBadIntervention = 1 - ((1 - pi) ** (numPenultimate) * (1 - initialQValues ** 2) +
                                        (1 - pi) ** (numPenultimate - 1) * (1 - pi - epsilon) * (initialQValues ** 2))
        regretForBadIntervention = probOfOneGoodIntervention - probOfOneBadIntervention

        for j in range(len(methodsTuple)):
            methodTuple = methodsTuple[j]
            method = methodTuple[0]
            numSamples = methodTuple[1]
            numExperimentsToAvgOver = methodTuple[2]
            regretMean, regretList = \
                getAvgRegret(experimentFunction=method, numTotalSamples=numSamples,
                             numExperimentsToAvgOver=numExperimentsToAvgOver, numPenultimate=numPenultimate,
                             degree=degree, pi=pi, epsilon=epsilon, probOf1AtLeaves=initialQValues)
            regretCompiled[i, j] = regretMean

        print("numPenultimate=", numPenultimate,"maxRegret=",regretForBadIntervention, "regretCompiled[i]=", regretCompiled[i])

    print("regretCompiled=", regretCompiled)

    # convert array into dataframe for saving
    dataFrame = pd.DataFrame(regretCompiled)
    colNames = [(methodTuple[0].__name__).replace("get", "") for methodTuple in methodsTuple]
    dataFrame.columns = colNames
    dataFrame.index = sorted(list(numNodes))
    # save the dataframe as a csv file
    filePathToSave = 'outputs/regretWithNumNodes_' + str(round(pi, 2)) + 'pi' + str(epsilon) + 'eps' + \
                     str(degree) + 'degree' + ''.join(colNames) + '.csv'
    dataFrame.to_csv(filePathToSave)

    print("time taken in seconds:", time.time() - startTime)

    # methodsTuple = [(getPureBanditRegret,1e6,10000),(getYabeRegret,2500,1000),(getCIRegret,300,10000)]
    # methodsTuple = [(getPureBanditRegret,1e6,20),(getYabeRegret,2500,20),(getCIRegret,300,20)]
    # methodsTuple = [(getPureBanditRegret,1e6,20),(getYabeRegret,2500,20),(getCIRegret,300,20)]
    # methodsTuple = [(getPureBanditRegret,2e4,1000)]
    # methodsTuple = [(getPureBanditRegret,3e4,1000)]
    # methodsTuple = [(getYabeRegret,5000,100)]
    # methodsTuple = [(getCIRegret,100,1000)]
    # methodsTuple = [(getPureBanditRegret, 3e4, 10000), (getYabeRegret, 5000, 1000), (getCIRegret, 100, 10000)]
    #
    # regretCompiled = np.zeros((len(listOfNumberOfLayers), len(methodsTuple)))

    # methods, degree, pi, epsilon, initialQValues, numTotalSamples, listOfNumberOfLayers, numExperimentsToAvgOver = \
    #     [getYabeRegret], 2, 0.2, 0.2, 0, 2500, list(range(3, 8)), 100
    # methods, degree, pi, epsilon, initialQValues, numTotalSamples, listOfNumberOfLayers, numExperimentsToAvgOver = \
    #     [getYabeRegret], 2, 0.1, 0.2, 0, 2000, list(range(3, 8)), 1000

    # methods, degree, pi, epsilon, initialQValues, numTotalSamples, listOfNumberOfLayers, numExperimentsToAvgOver = \
    #     [getCIRegret], 2, 0.1, 0.2, 0, 250, list(range(3, 8)), 10000

    # degree, pi, epsilon, methods, numTotalSamples = 2, 0.25, 0.125, [getCIRegret], 1000
    # numNodes = set()
    # # regretCompiled = np.zeros((len(listOfNumberOfLayers), len(methods)))
    # cgraph = CoveredGraph.__new__(CoveredGraph)
    # cgraph.__init__(degree=degree, numLayers=listOfNumberOfLayers[0], initialQValues=initialQValues, pi=pi,
    #                 epsilon=epsilon)
    # for i in range(len(listOfNumberOfLayers)):
    #     numLayers = listOfNumberOfLayers[i]
    #     # pi = 0.01/cgraph.numNodes
    #     # print("pi=", pi)
    #     cgraph = CoveredGraph.__new__(CoveredGraph)
    #     cgraph.__init__(degree=degree, numLayers=listOfNumberOfLayers[i], initialQValues=initialQValues, pi=pi,
    #                     epsilon=epsilon)
    #     numNodes.add(cgraph.numNodes)
    #
    #     for j in range(len(methodsTuple)):
    #         method = methodsTuple[j][0]
    #         numTotalSamples = methodsTuple[j][1]
    #         numExperimentsToAvgOver = methodsTuple[j][2]
    #
    #         regretMean, regretList = getAvgRegret(numExperimentsToAvgOver, method,
    #                                               numTotalSamples, degree, numLayers,
    #                                               initialQValues, pi, epsilon)
    #         # regretCompiled[i, j] = regretMean/cgraph.regretOnChoosingBadIntervention
    #         regretCompiled[i, j] = regretMean
    #     print("cgraph.regretOnChoosingBadIntervention=", cgraph.regretOnChoosingBadIntervention)
    #     print("numLayers=", numLayers, "regretCompiled[i]=", regretCompiled[i])
