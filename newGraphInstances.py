import numpy as np, scipy as sc, random as rd


class CoveredGraph:
    def __new__(cls, *args, **kwargs):
        # print("1. Create a new instance of the graph.")
        return super().__new__(cls)

    def __init__(self, num_vertices, degree, initialQValues=0.01, pi=0.05, epsilon=0.1):
        self.numVertices = num_vertices
        self.degree = degree
        # print("Initialize the new instance of graph.")
        self.graph, self.numOfParentsPerVertex = self.generateGraph(num_vertices, degree)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(degree={self.degree}, numVertices={self.numVertices}, " \
               f"\ngraph={self.graph}\n numOfParentsPerVertex={self.numOfParentsPerVertex}"

    def generateGraph(self, num_vertices, degree):
        g = np.ones((num_vertices, degree)) * -1
        numOfParentsPerVertex = np.zeros(num_vertices)
        for i in range(1, num_vertices):
            setOfParents = list(set(rd.choices(range(i), k=degree)))
            numParentsI = len(setOfParents)
            numOfParentsPerVertex[i] = numParentsI
            g[i][:numParentsI] = setOfParents

        return g, numOfParentsPerVertex


if __name__ == "__main__":
    graph = CoveredGraph(15, 3)
    print("graph=", graph)
