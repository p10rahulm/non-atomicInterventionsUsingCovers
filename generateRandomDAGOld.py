'''
@manual{squires2018causaldag,
  title={{\texttt{causaldag}: creation, manipulation, and learning of causal models}},
  author={{Chandler Squires}},
  year={2018},
  url={https://github.com/uhlerlab/causaldag},
}
'''

import numpy as np
import random as rd
import networkx as nx
import matplotlib.pyplot as plt
from causaldag import rand


def plot_DAG(myGraph,savePath,show=False):
    nx.draw_networkx(myGraph, arrows=True)
    plt.savefig(savePath, format="PNG")
    if show:
        plt.show()
    return plt.clf

if __name__=="__main__":
    np.random.seed(8)
    rd.seed(8)

    nnodes = 10
    dag = rand.directed_erdos(nnodes, .5)
    print("dag=", dag)
    nxGraph = dag.to_nx()
    savePath = "outputs/DAG.png"
    plot_DAG(nxGraph,savePath,True)

