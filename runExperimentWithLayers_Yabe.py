from pureBandit import getPureBanditRegret
from Yabe import getYabeRegret
from CoveringInterventions import getCIRegret
from utils import getAvgRegret
import time,numpy as np, random as rd, statistics as stats, pandas as pd
from coveredTree import CoveredGraph, randomBool
from tqdm import tqdm


if __name__ == "__main__":
    startTime = time.time()
    np.random.seed(8)
    np.set_printoptions(precision=3)
    rd.seed(8)

    degree, numLayers, initialQValues, pi, epsilon = 2, 3, 0, 0.25, 0.125
    # numTotalSamples = 2000
    numExperimentsToAvgOver = 50
    # methods = [getPureBanditRegret,getYabeRegret,getCIRegret]
    # methods = [getCIRegret]
    degree, pi,epsilon, methods,numTotalSamples = 2, 0.25,0.125,[getYabeRegret],10000
    # degree, pi,epsilon, methods,numTotalSamples = 2, 0.25,0.125,[getCIRegret],10000

    # listOfNumberOfLayers = list(range(3,11))
    listOfNumberOfLayers = list(range(3,8))
    numNodes = set()
    regretCompiled = np.zeros((len(listOfNumberOfLayers),len(methods)))
    for i in range(len(listOfNumberOfLayers)):
        for j in range(len(methods)):
            numLayers = listOfNumberOfLayers[i]
            cgraph = CoveredGraph.__new__(CoveredGraph)
            cgraph.__init__(degree=degree, numLayers=numLayers, initialQValues=initialQValues, pi=pi, epsilon=epsilon)
            numNodes.add(cgraph.numNodes)
            method = methods[j]
            regretMean, regretList = getAvgRegret(numExperimentsToAvgOver, method,
                                                  numTotalSamples, degree, numLayers,
                                                  initialQValues, pi, epsilon)
            regretCompiled[i,j] = regretMean
        print("regretCompiled[i]=",regretCompiled[i])

    print("regretCompiled=", regretCompiled)

    # convert array into dataframe for saving
    dataFrame = pd.DataFrame(regretCompiled)
    colNames = [(method.__name__).replace("get", "") for method in methods]
    dataFrame.columns = colNames
    dataFrame.index = sorted(list(numNodes))
    # save the dataframe as a csv file
    filePathToSave = 'outputs/regretWithNumLayers_' + str(numTotalSamples) + 'obs' + ''.join(colNames) + '.csv'
    dataFrame.to_csv(filePathToSave)

    print("time taken in seconds:", time.time() - startTime)
