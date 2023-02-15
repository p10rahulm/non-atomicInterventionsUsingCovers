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

    # degree, numLayers, initialQValues, mu, epsilon = 2, 3, 0, 0.125, 0.125
    # numTotalSamples = 2000
    # numExperimentsToAvgOver = 1000
    # methods = [getPureBanditRegret,getYabeRegret,getCIRegret]
    # methods = [getCIRegret]

    methods, degree, mu, epsilon, initialQValues, numTotalSamples, listOfNumberOfLayers, numExperimentsToAvgOver = \
        [getPureBanditRegret], 2, 0.1, 0.2, 0, 1000000, list(range(3, 8)), 100


    # methods, degree, mu, epsilon, initialQValues, numTotalSamples, listOfNumberOfLayers, numExperimentsToAvgOver = \
    #     [getYabeRegret], 2, 0.2, 0.2, 0, 2500, list(range(3, 8)), 100
    # methods, degree, mu, epsilon, initialQValues, numTotalSamples, listOfNumberOfLayers, numExperimentsToAvgOver = \
    #     [getYabeRegret], 2, 0.1, 0.2, 0, 2000, list(range(3, 8)), 1000

    # methods, degree, mu, epsilon, initialQValues, numTotalSamples, listOfNumberOfLayers, numExperimentsToAvgOver = \
    #     [getCIRegret], 2, 0.1, 0.2, 0, 250, list(range(3, 8)), 10000

    # degree, mu, epsilon, methods, numTotalSamples = 2, 0.25, 0.125, [getCIRegret], 1000
    numNodes = set()
    regretCompiled = np.zeros((len(listOfNumberOfLayers), len(methods)))
    for i in range(len(listOfNumberOfLayers)):
        for j in range(len(methods)):
            numLayers = listOfNumberOfLayers[i]
            cgraph = CoveredGraph.__new__(CoveredGraph)
            cgraph.__init__(degree=degree, numLayers=numLayers, initialQValues=initialQValues, mu=mu, epsilon=epsilon)
            numNodes.add(cgraph.numNodes)
            method = methods[j]
            regretMean, regretList = getAvgRegret(numExperimentsToAvgOver, method,
                                                  numTotalSamples, degree, numLayers,
                                                  initialQValues, mu, epsilon)
            regretCompiled[i, j] = regretMean
        print("regretCompiled[i]=", regretCompiled[i])

    print("regretCompiled=", regretCompiled)

    # convert array into dataframe for saving
    dataFrame = pd.DataFrame(regretCompiled)
    colNames = [(method.__name__).replace("get", "") for method in methods]
    dataFrame.columns = colNames
    dataFrame.index = sorted(list(numNodes))
    # save the dataframe as a csv file
    filePathToSave = 'outputs/regretWithNumLayers_' + str(mu) + 'mu' + str(epsilon) + 'eps' + \
                     str(numTotalSamples) + 'obs' + ''.join(colNames) + '.csv'
    dataFrame.to_csv(filePathToSave)

    print("time taken in seconds:", time.time() - startTime)
