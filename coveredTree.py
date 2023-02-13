import numpy as np, random as rd


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
    #   2. all internal nodes (except the last) in the penultimate layer take value 1 with probability 0.5
    #   3. the last node in the penultimate layer takes value:
    #       3a. 1 with probability 0.5 + epsilon if all its parents are set to 1.
    #       3b. 1 with probability 0.5 otherwise.
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
        print("1. Create a new instance of the graph.")
        return super().__new__(cls)

    def __init__(self, degree, numLayers, initialQValues=0.01, chosenEpsilon=0.3):
        print("2. Initialize the new instance of Point.")
        self.degree = degree  # number of edges per node.
        self.numLayers = numLayers  # Root node is considered layer 1. So a 2 layer tree has only leaves and root
        self.numNodes = int((degree ** (numLayers + 1) - 1) / (degree - 1))
        self.numNodes = int((degree ** (numLayers + 1) - 1) / (degree - 1))
        self.numLeaves = int(degree ** (numLayers))
        self.numInternal = self.numNodes - self.numLeaves
        self.numPenultimate = int(degree ** (numLayers - 1))
        self.chosenEpsilon = chosenEpsilon
        self.leafQvals = np.ones(self.numLeaves) * initialQValues

    def __repr__(self) -> str:
        return f"{type(self).__name__}(degree={self.degree}, numLayers={self.numLayers}, " \
               f"numNodes={self.numNodes}, numLeaves={self.numLeaves}, , numInternal={self.numInternal}" \
               f"\nnumPenultimate={self.numPenultimate}, chosenEpsilon={self.chosenEpsilon}" \
               f"\nleafQvals={self.leafQvals})"

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
        for i in reversed(range(self.numNodes)):
            if self.checkLeafIndex(i):
                leafIndex = i-self.numInternal
                qVal = self.leafQvals[leafIndex]
                sampledVals[i] = randomBool(qVal)
            if self.checkPenultimateLayer(i):
                if self.checkChosenParentPenultimate(i):
                    qVal = 0.5 + self.leafQvals[i]
                    sampledVals[i] = randomBool(qVal)
                else:
                    qVal = 0.5
                    sampledVals[i] = randomBool(qVal)
            else:
                childIndices = self.getChildIndices(i)
                # sum = 0
                # for j in childIndices:
                #     sum += sampledVals[j]
                # sampledVals[i] = sum
                sampledVals[i] = sampledVals[childIndices].sum()


        return sampledVals


if __name__ == "__main__":
    np.random.seed(8)
    rd.seed(8)
    cgraph = CoveredGraph.__new__(CoveredGraph)
    cgraph.__init__(degree=3, numLayers=4, initialQValues=0.0)
    print("cgraph=", cgraph)
    print("cgraph.numNodes=", cgraph.numNodes)
    for i in range(cgraph.numNodes):
        print("child Node = %d, parentNode = %d" % (i, cgraph.getParentIndex(i)))

    for i in range(cgraph.numNodes):
        print("parent Node = %d, child Nodes = %s" % (i, cgraph.getChildIndices(i)))
    # cg2 = CoveredGraph(degree=3, numLayers=4,initialQValues=0.01)
    # print("cg2=", cg2)
    for i in range(5):
        print("cgraph Values on do nothing =", cgraph.doNothing())
