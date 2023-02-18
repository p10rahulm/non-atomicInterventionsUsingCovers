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
    # cgraph.__init__(degree=3, numLayers=4, initialQValues=0.0,pi=0.05,epsilon=0.05)
    degree, numLayers, initialQValues, pi, epsilon = 2, 3, 0, 0.1, 0.2
    # numTotalSamples = 2000
    numExperimentsToAvgOver = 50
    methods = [getPureBanditRegret,getYabeRegret,getCIRegret]
    methods = [getPureBanditRegret]
    numSamplesToChoose = [500,1000,2000,4000,8000,16000,32000,64000,128000]
    regretCompiled = np.zeros((len(numSamplesToChoose),len(methods)))
    for i in range(len(numSamplesToChoose)):
        for j in range(len(methods)):
            numTotalSamples = numSamplesToChoose[i]
            method = methods[j]
            regretMean, regretList = getAvgRegret(numExperimentsToAvgOver, method,
                                                  numTotalSamples, degree, numLayers,
                                                  initialQValues, pi, epsilon)
            regretCompiled[i,j] = regretMean

    print("regretCompiled=", regretCompiled)

    # convert array into dataframe for saving
    filePathToSave = 'outputs/regretWithT.csv'
    dataFrame = pd.DataFrame(regretCompiled)
    dataFrame.columns = [(method.__name__).replace("get", "") for method in methods]
    dataFrame.index = numSamplesToChoose
    # save the dataframe as a csv file
    dataFrame.to_csv(filePathToSave)

    print("time taken in seconds:", time.time() - startTime)
