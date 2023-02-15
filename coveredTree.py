import numpy as np, random as rd
import statistics as stats
import time, collections
import math


def randomBool(qVal):
    # choice = np.random.choice([0, 1], p=[qVal, 1 - qVal])
    choice = np.random.binomial(n=1, p=qVal)
    return choice


class CoveredGraph:
    # We consider complete trees of k layers (k ≥ 2) and degree d (d ≥ 2)
    # The chosen intervention is the values {1,1,...,d} at the parents of the 1st node in the penultimate layer
    # The SCM is that:
    #   1. each of the internal nodes which are not in the penultimate layer are simply sum variables
    #   AUTHOR COMMENT:
    #   Earlier we chose 'OR' for these variables, but in large networks, it almost always gives value 1 at the output.
    #   2. all internal nodes (except the last) in the penultimate layer take value 1 with probability mu
    #   3. the last node in the penultimate layer takes value:
    #       3a. 1 with probability mu + epsilon if all its parents are set to 1.
    #       3b. 1 with probability mu otherwise.
    #
    # The leaf nodes are boolean variables that take value 1 as per the q values
    #
    # We will use the array representation so that we can easily find the parent node of any given node.
    # For example:
    #   1. Reward node is node at index 0
    #   2. The children of the reward node are variables with index 1-d
    #   3. The child of node 1 is at index d + 1
    #
    # In general, the parent of node at index k is the node with index m = ceil(k/d) + 1

    def __new__(cls, *args, **kwargs):
        # print("1. Create a new instance of the graph.")
        return super().__new__(cls)

    def __init__(self, degree, numLayers, initialQValues=0.01, mu=0.05, epsilon=0.1):
        # print("Initialize the new instance of graph.")
        self.degree = degree  # number of edges per node.
        self.numLayers = numLayers  # Root node is considered layer 1. So a 2 layer tree has only leaves and root
        self.numNodes = int((degree ** (numLayers + 1) - 1) / (degree - 1))
        self.numNodes = int((degree ** (numLayers + 1) - 1) / (degree - 1))
        self.numLeaves = int(degree ** (numLayers))
        self.numInternal = self.numNodes - self.numLeaves
        self.numPenultimate = int(degree ** (numLayers - 1))

        self.assignmentSet = self.getAssignmentSet()
        self.leafNodeStartIndex = self.numNodes - self.numLeaves  # inclusive
        self.leafNodeEndIndex = self.numNodes - 1  # inclusive
        self.penultimateLayerStartIndex = self.leafNodeStartIndex - self.numPenultimate  # inclusive
        self.penultimateLayerEndIndex = self.leafNodeStartIndex - 1  # inclusive
        self.leafSet = list(range(self.leafNodeStartIndex, self.leafNodeEndIndex + 1))
        self.penultimateLayerSet = list(range(self.penultimateLayerStartIndex, self.penultimateLayerEndIndex + 1))
        self.chosenInterventionSet = self.getChildIndices(self.penultimateLayerEndIndex)
        self.chosenInterventionValues = [1]*self.degree

        self.mu = mu
        self.epsilon = epsilon
        self.probOf1AtLeaves = initialQValues
        self.leafQvals = np.ones(self.numLeaves) * self.probOf1AtLeaves

    def __repr__(self) -> str:
        return f"{type(self).__name__}(degree={self.degree}, numLayers={self.numLayers}, " \
               f"\nnumNodes={self.numNodes}, numLeaves={self.numLeaves}, numInternal={self.numInternal}" \
               f"\nleafNodeStartIndex={self.leafNodeStartIndex}, leafNodeEndIndex={self.leafNodeEndIndex}, " \
               f"\npenultimateLayerStartIndex={self.penultimateLayerStartIndex}, " \
               f"penultimateLayerEndIndex={self.penultimateLayerEndIndex}, " \
               f"\nleafSet={self.leafSet}, penultimateLayerSet={self.penultimateLayerSet}, " \
               f"\nnumPenultimate={self.numPenultimate}, mu={self.mu}, epsilon={self.epsilon}" \
               f"\nchosenInterventionSet={self.chosenInterventionSet}, " \
               f"chosenInterventionValues={self.chosenInterventionValues}" \
               f"\nleafQvals={self.leafQvals}), assignmentSet={self.assignmentSet})"

    def checkLeafIndex(self, k):
        if k < 0 or k >= self.numNodes:
            return "Not a node index"
        if k >= (self.numNodes - self.numLeaves):
            return 1
        return 0

    def checkPenultimateLayer(self, k):
        if k < 0 or k >= self.numNodes:
            return -1
        if (self.numNodes - self.numLeaves - 1) >= k >= (self.numNodes - self.numLeaves - self.numPenultimate):
            return 1
        return 0

    def checkChosenParentPenultimate(self, k):
        if k < 0 or k >= self.numNodes:
            return -1
        if k == (self.numNodes - self.numLeaves - 1):
            return 1
        return 0

    def getParentIndex(self, k):
        if k < 0 or k >= self.numNodes:
            return "Not a node index"
        if k == 0:
            return -1
        parentIndex = np.ceil(k / self.degree) - 1
        return parentIndex

    def getChildIndices(self, k):
        if self.checkLeafIndex(k):
            return -1
        startIndex = self.degree * k + 1
        endIndex = self.degree * (k + 1)
        childrenIndices = list(range(startIndex, endIndex + 1))
        return childrenIndices

    def doNothing(self):
        sampledVals = np.zeros(self.numNodes)
        for currentIndex in reversed(range(self.numNodes)):
            if self.checkLeafIndex(currentIndex):
                leafIndex = currentIndex - self.numInternal
                qVal = self.leafQvals[leafIndex]
                sampledVals[currentIndex] = randomBool(qVal)
            elif self.checkPenultimateLayer(currentIndex):
                if self.checkChosenParentPenultimate(currentIndex):
                    childIndices = self.getChildIndices(currentIndex)
                    if sampledVals[childIndices].sum() == self.degree:
                        qVal = self.mu + self.epsilon
                    else:
                        qVal = self.mu
                    sampledVals[currentIndex] = randomBool(qVal)
                else:
                    qVal = self.mu
                    sampledVals[currentIndex] = randomBool(qVal)
            else:
                childIndices = self.getChildIndices(currentIndex)
                # sum = 0
                # for j in childIndices:
                #     sum += sampledVals[j]
                # sampledVals[i] = sum
                sampledVals[currentIndex] = (sampledVals[childIndices].sum() > 0) * 1
        return sampledVals

    def getRewardOnDoNothing(self):
        # Check if any of the last but 1 in the penultimate layer is 1, then return reward 1
        for i in range(self.numPenultimate - 1):
            if randomBool(self.mu) == 1:
                return 1
        # Check if all children of the last penultimate layer node is 1
        topBool = 1
        for i in range(self.degree):
            if randomBool(self.mu) == 0:
                topBool = 0
        if topBool:
            return randomBool(self.mu + self.epsilon)
        else:
            return randomBool(self.mu)

    def doOperation(self, intervenedIndices, intervenedValues):
        for currentIndex in intervenedIndices:
            if not self.checkLeafIndex(currentIndex):
                raise Exception("Sorry, not a valid do() operation")
        for value in intervenedValues:
            if value not in [0, 1]:
                raise Exception("Sorry, not a valid do() operation")

        sampledVals = np.zeros(self.numNodes)
        for currentIndex in reversed(range(self.numNodes)):
            if self.checkLeafIndex(currentIndex):
                if currentIndex in intervenedIndices:
                    currentIndexLocationInInterevenedArray = intervenedIndices.index(currentIndex)
                    intervenedValue = intervenedValues[currentIndexLocationInInterevenedArray]
                    sampledVals[currentIndex] = intervenedValue
                else:
                    leafIndex = currentIndex - self.numInternal
                    qVal = self.leafQvals[leafIndex]
                    sampledBoolean = randomBool(qVal)
                    sampledVals[currentIndex] = sampledBoolean

            elif self.checkPenultimateLayer(currentIndex):
                if self.checkChosenParentPenultimate(currentIndex):
                    childIndices = self.getChildIndices(currentIndex)
                    if sampledVals[childIndices].sum() == self.degree:
                        qVal = self.mu + self.epsilon
                    else:
                        qVal = self.mu
                    sampledVals[currentIndex] = randomBool(qVal)
                else:
                    qVal = self.mu
                    sampledVals[currentIndex] = randomBool(qVal)
            else:
                childIndices = self.getChildIndices(currentIndex)
                sampledVals[currentIndex] = (sampledVals[childIndices].sum() > 0) * 1
        return sampledVals

    def sampleLeaves(self, numSamples):
        # print("numSamples=",numSamples)
        probArray = []
        for i in range(self.degree+1):
            numTimes = math.comb(self.degree, i)
            value = (1 - self.probOf1AtLeaves) ** (self.degree-i) * self.probOf1AtLeaves ** i
            probArray.extend([value]*numTimes)
        # print("probarray=",probArray)
        choiceArray = np.random.choice(2**self.degree, numSamples, p=probArray)
        # print("choiceArray=", choiceArray)
        countOfChoicesDict = collections.Counter(choiceArray)
        # print("countOfChoicesDict=", countOfChoicesDict)
        countOfChoicesOrderedArray=[]
        for i in range(2**self.degree):
            countOfChoicesOrderedArray.append(countOfChoicesDict[i])
        # print("countOfChoicesOrderedArray=", countOfChoicesOrderedArray)
        # countOfChoicesOrderedArray = [countOfChoicesDict[x] for x in sorted(countOfChoicesDict.keys())]
        return countOfChoicesOrderedArray


    def getProbsWithDoOperation(self, opPenultimateIndex, opPenultimateValueIndex, numTimesIntervened):
        if opPenultimateIndex < 0 or opPenultimateValueIndex >= self.numPenultimate or \
                opPenultimateValueIndex < 0 or opPenultimateValueIndex >= 2 ** self.degree:
            raise Exception("Sorry, not a valid do() operation")
        numTimesSeen = np.zeros((self.numPenultimate, 2 ** self.degree))
        numTimesOne = np.zeros((self.numPenultimate, 2 ** self.degree))
        for penultimateIndex in range(self.numPenultimate):
            if penultimateIndex!=opPenultimateIndex:
                # print("opPenultimateIndex=",opPenultimateIndex,"opPenultimateValueIndex=",opPenultimateValueIndex)
                returnedSamples = self.sampleLeaves(numTimesIntervened)
                # print("numTimesIntervened=", numTimesIntervened,"returnedSamples=",returnedSamples)
                numTimesSeen[penultimateIndex] = returnedSamples
            else:
                numTimesSeen[penultimateIndex,opPenultimateValueIndex] = numTimesIntervened

        for row in range(self.numPenultimate):
            for col in range(2**self.degree):
                if row == self.numPenultimate-1 and col == 2**self.degree-1:
                    numTimesOne[row, col] = np.random.binomial(n=numTimesSeen[row, col], p=self.mu+self.epsilon)
                else:
                    numTimesOne[row,col] = np.random.binomial(n=numTimesSeen[row,col], p=self.mu)

        return numTimesSeen,numTimesOne

    def sampleLeavesUniformly(self,numSamples):
        choiceArray = np.random.choice(2 ** self.degree, numSamples)
        countOfChoicesDict = collections.Counter(choiceArray)
        # print("countOfChoicesDict=", countOfChoicesDict)
        countOfChoicesOrderedArray = []
        for i in range(2 ** self.degree):
            countOfChoicesOrderedArray.append(countOfChoicesDict[i])

        return countOfChoicesOrderedArray

    def getProbsWithDoOperationUsingCovers(self, numTimesIntervened):
        if numTimesIntervened < 0:
            raise Exception("Sorry, not a valid do() operation")
        numTimesSeen = np.zeros((self.numPenultimate, 2 ** self.degree))
        numTimesOne = np.zeros((self.numPenultimate, 2 ** self.degree))
        for penultimateIndex in range(self.numPenultimate):
            numTimesSeen[penultimateIndex] = self.sampleLeavesUniformly(numTimesIntervened)

        for row in range(self.numPenultimate):
            for col in range(2**self.degree):
                if row == self.numPenultimate-1 and col == 2**self.degree-1:
                    numTimesOne[row, col] = np.random.binomial(n=numTimesSeen[row, col], p=self.mu+self.epsilon)
                else:
                    numTimesOne[row,col] = np.random.binomial(n=numTimesSeen[row,col], p=self.mu)

        return numTimesSeen,numTimesOne

    def getRewardOnDoOperation(self, intervenedIndices, intervenedValues):
        if np.array_equal(intervenedIndices, self.chosenInterventionSet) and \
                np.array_equal(intervenedValues, self.chosenInterventionValues):
            # If chosen intervention is done, and the boolean value is 1, then return 1
            if randomBool(self.mu + self.epsilon):
                return 1
            # Or if any of the other values are randomly known to be 1, then return 1
            else:
                if np.random.binomial(n=self.numPenultimate - 1, p=self.mu) > 0:
                    return 1

        else:
            # But if chosen intervention is not 1, but any of the other booleans is randomly 1, then return 1
            if np.random.binomial(n=self.numPenultimate, p=self.mu) > 0:
                return 1
        # if none of the random values is 1, return 0
        return 0

    def getAvgRewardOnMultipleDoOperations(self, intervenedIndices, intervenedValues, numOps):
        if np.array_equal(intervenedIndices, self.chosenInterventionSet) and \
                np.array_equal(intervenedValues,self.chosenInterventionValues):
            # If chosen intervention is done:
            # Then prob of 1 is 1-prob of 0 = 1- prob that all penultimate nodes are 0
            probOf1 = 1 - (1 - self.mu - self.epsilon) * (1 - self.mu) ** (self.numPenultimate - 1)
            num1s = np.random.binomial(n=numOps, p=probOf1)
            avg = num1s / numOps
            # print("num1s=", num1s, "avg=", avg)
        else:
            # But if chosen intervention is not the intervened one
            # Then prob of 1 is 1-prob of 0 = 1- prob that all penultimate nodes are 0
            probOf1 = 1 - ((1 - self.mu) ** (self.numPenultimate) * (1 - self.probOf1AtLeaves ** 2) +
                           (1 - self.mu) ** (self.numPenultimate - 1) * (1 - self.mu - self.epsilon) * (
                                   self.probOf1AtLeaves ** 2))

            num1s = np.random.binomial(n=numOps, p=probOf1)
            avg = num1s / numOps
            # print("num1s=",num1s,"avg=",avg)

        return avg

    def getAssignmentSet(self):
        stringFormat = '{0:0' + str(self.degree) + 'b}'
        assignments = [[int(j) for j in stringFormat.format(i)] for i in range(2 ** self.degree)]
        sortedAssignments = sorted(assignments, key=sum)
        return sortedAssignments


    def getInterventionFromIndex(self, index):
        penultimateIndex = self.penultimateLayerStartIndex + index // (2 ** self.degree)
        childrenOfPenultimate = self.getChildIndices(penultimateIndex)
        assignmentIndex = index % (2 ** self.degree)
        assignment = self.assignmentSet[assignmentIndex]
        return (penultimateIndex, childrenOfPenultimate, assignmentIndex, assignment)


if __name__ == "__main__":
    startTime = time.time()
    np.random.seed(8)
    np.set_printoptions(precision=3)
    # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    rd.seed(8)
    cgraph = CoveredGraph.__new__(CoveredGraph)
    cgraph.__init__(degree=3, numLayers=4, initialQValues=0.0, mu=0.05, epsilon=0.05)
    print("cgraph=", cgraph)
    print("cgraph.numNodes=", cgraph.numNodes)
    for i in range(cgraph.numNodes):
        print("child Node = %d, parentNode = %d" % (i, cgraph.getParentIndex(i)))

    for i in range(cgraph.numNodes):
        print("parent Node = %d, child Nodes = %s" % (i, cgraph.getChildIndices(i)))

    graphObs = cgraph.doNothing()
    print("graphObs=", graphObs)
    rewards = []
    numIters = 100
    for i in range(numIters):
        graphObs = cgraph.doNothing()
        rewards.append(graphObs[0])
    # print("cgraph Values on do nothing =", rewards)
    print("cgraph Average Rewards on Do nothing =", stats.fmean(rewards))
    rewardsOnDo = []
    for i in range(numIters):
        graphObs = cgraph.doOperation([118, 119, 120], [1, 1, 1])
        rewardsOnDo.append(graphObs[0])
    # print("cgraph Values on do lastNodes =", rewardsOnDo)
    print("cgraph Average Rewards on Do =", stats.fmean(rewardsOnDo))
    assignments = cgraph.getAssignmentSet()
    print("assignments=", assignments)

    print("cgraph=", cgraph)
    print("time taken in seconds:", time.time() - startTime)
