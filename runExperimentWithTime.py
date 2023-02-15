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
    # cgraph = CoveredGraph.__new__(CoveredGraph)
    # cgraph.__init__(degree=3, numLayers=4, initialQValues=0.0,mu=0.05,epsilon=0.05)
    # degree, numLayers, initialQValues, mu, epsilon = 3, 3, 0, 0.1, 0.05
    degree, numLayers, initialQValues, mu, epsilon = 3, 3, 0, 0.2, 0.2
    # degree, numLayers, initialQValues, mu, epsilon = 3, 6, 0, 0.1, 0.05
    # numTotalSamples = 2000
    numExperimentsToAvgOver = 1000
    methods = [getPureBanditRegret,getYabeRegret,getCIRegret]
    # methods = [getPureBanditRegret]
    numSamplesToChoose = [100,250,500,1000,2500,5000,10000,25000,50000,100000]
    regretCompiled = np.zeros((len(numSamplesToChoose),len(methods)))
    for i in range(len(numSamplesToChoose)):
        for j in range(len(methods)):
            numTotalSamples = numSamplesToChoose[i]
            method = methods[j]
            regretMean, regretList = getAvgRegret(numExperimentsToAvgOver, method,
                                                  numTotalSamples, degree, numLayers,
                                                  initialQValues, mu, epsilon)
            regretCompiled[i,j] = regretMean
        print("regretCompiled[i]=", regretCompiled[i])
    print("regretCompiled=", regretCompiled)

    # convert array into dataframe for saving
    colNames = [(method.__name__).replace("get", "") for method in methods]
    dataFrame = pd.DataFrame(regretCompiled)
    dataFrame.columns = colNames
    dataFrame.index = numSamplesToChoose
    # save the dataframe as a csv file
    filePathToSave = 'outputs/regretWithT_' + str(mu) + 'mu' + str(epsilon) + 'eps' + \
                     str(numTotalSamples) + 'obs' + ''.join(colNames) + '.csv'
    dataFrame.to_csv(filePathToSave)

    print("time taken in seconds:", time.time() - startTime)
