from pureBandit import getPureBanditRegret, getPureBanditRegret2
from Yabe import getYabeRegret
from CoveringInterventions import getCIRegret
from utils import getAvgRegret
import time, numpy as np, random as rd, statistics as stats, pandas as pd
from coveredTree import CoveredGraph, randomBool
from tqdm import tqdm

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
    degree,height, pi, epsilon, initialQValues, numExperimentsToAvgOver = 2,3, 0.001, 0.05, 0, 10000
    degrees=list(range(2, 9))

    # methodsTuple = [(getPureBanditRegret,1e6,10000),(getYabeRegret,2500,1000),(getCIRegret,300,10000)]
    # methodsTuple = [(getPureBanditRegret,1e6,20),(getYabeRegret,2500,20),(getCIRegret,300,20)]
    # methodsTuple = [(getPureBanditRegret,1e6,20),(getYabeRegret,2500,20),(getCIRegret,300,20)]
    # methodsTuple = [(getPureBanditRegret,2e4,1000)]
    # methodsTuple = [(getPureBanditRegret,3e4,1000)]
    # methodsTuple = [(getYabeRegret,5000,100)]
    # methodsTuple = [(getCIRegret,100,1000)]
    methodsTuple = [(getPureBanditRegret,3e4,10000),(getYabeRegret,5000,1000),(getCIRegret,100,10000)]
    methodsTuple = [(getPureBanditRegret,5e5,100)]


    regretCompiled = np.zeros((len(degrees),len(methodsTuple)))

    # methods, degree, pi, epsilon, initialQValues, numTotalSamples, listOfNumberOfLayers, numExperimentsToAvgOver = \
    #     [getYabeRegret], 2, 0.2, 0.2, 0, 2500, list(range(3, 8)), 100
    # methods, degree, pi, epsilon, initialQValues, numTotalSamples, listOfNumberOfLayers, numExperimentsToAvgOver = \
    #     [getYabeRegret], 2, 0.1, 0.2, 0, 2000, list(range(3, 8)), 1000

    # methods, degree, pi, epsilon, initialQValues, numTotalSamples, listOfNumberOfLayers, numExperimentsToAvgOver = \
    #     [getCIRegret], 2, 0.1, 0.2, 0, 250, list(range(3, 8)), 10000

    # degree, pi, epsilon, methods, numTotalSamples = 2, 0.25, 0.125, [getCIRegret], 1000
    numNodes = set()
    # regretCompiled = np.zeros((len(listOfNumberOfLayers), len(methods)))
    cgraph = CoveredGraph.__new__(CoveredGraph)
    cgraph.__init__(degree=degree, numLayers=height, initialQValues=initialQValues, pi=pi,
                    epsilon=epsilon)
    for i in range(len(degrees)):
        degree = degrees[i]
        # pi = 0.01/cgraph.numNodes
        # print("pi=", pi)
        cgraph = CoveredGraph.__new__(CoveredGraph)
        cgraph.__init__(degree=degree, numLayers=height, initialQValues=initialQValues, pi=pi,
                        epsilon=epsilon)
        numNodes.add(cgraph.numNodes)

        for j in range(len(methodsTuple)):
            method = methodsTuple[j][0]
            numTotalSamples = methodsTuple[j][1]
            numExperimentsToAvgOver = methodsTuple[j][2]

            regretMean, regretList = getAvgRegret(numExperimentsToAvgOver, method,
                                                  numTotalSamples, degree, height,
                                                  initialQValues, pi, epsilon)
            # regretCompiled[i, j] = regretMean/cgraph.regretOnChoosingBadIntervention
            regretCompiled[i, j] = regretMean
        print("cgraph.regretOnChoosingBadIntervention=",cgraph.regretOnChoosingBadIntervention)
        print("degree=",degree,"regretCompiled[i]=", regretCompiled[i])

    print("regretCompiled=", regretCompiled)

    # convert array into dataframe for saving
    dataFrame = pd.DataFrame(regretCompiled)
    colNames = [(methodTuple[0].__name__).replace("get", "") for methodTuple in methodsTuple]
    dataFrame.columns = colNames
    dataFrame.index = sorted(list(numNodes))
    # save the dataframe as a csv file
    filePathToSave = 'outputs/regretWithDegree_' + str(round(pi,2)) + 'pi' + str(epsilon) + 'eps' + \
                     str(numTotalSamples) + 'obs' + ''.join(colNames) + '.csv'
    dataFrame.to_csv(filePathToSave)

    print("time taken in seconds:", time.time() - startTime)
