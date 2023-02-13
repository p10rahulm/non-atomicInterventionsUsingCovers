import random as rd, math, argparse
import numpy as np
from numpy.random.mtrand import sample
from matplotlib import patches, pyplot as plt
import networkx as nx
from scipy import sparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='default', type=str)  # parameters setting
parser.add_argument('--n', default=10, type=int)  # number of DAG  nodes
parser.add_argument('--max_out', default=2, type=float)  # max out_degree of one node
parser.add_argument('--alpha', default=1, type=float)  # shape
parser.add_argument('--beta', default=1.0, type=float)  # regularity
args = parser.parse_args()

set_dag_size = [20, 30, 40, 50, 60, 70, 80, 90]  # random number of DAG  nodes
set_max_out = [1, 2, 3, 4, 5]  # max out_degree of one node
set_alpha = [0.5, 1.0, 1.5]  # DAG shape
set_beta = [0.0, 0.5, 1.0, 2.0]  # DAG regularity


def DAGs_generate(mode='default', n=10, max_out=2, alpha=1, beta=1.0):
    ##############################################initialize############################################
    if mode != 'default':
        args.n = rd.sample(set_dag_size, 1)[0]
        args.max_out = rd.sample(set_max_out, 1)[0]
        args.alpha = rd.sample(set_alpha, 1)[0]
        args.beta = rd.sample(set_alpha, 1)[0]
    else:
        args.n = n
        args.max_out = max_out
        args.alpha = alpha
        args.beta = beta
        args.prob = 1

    length = math.floor(math.sqrt(args.n) / args.alpha)
    mean_value = args.n / length
    random_num = np.random.normal(loc=mean_value, scale=args.beta, size=(length, 1))
    ###############################################division#############################################
    position = {'Start': (0, 4), 'Exit': (10, 4)}
    generate_num = 0
    dag_num = 1
    dag_list = []
    for i in range(len(random_num)):
        dag_list.append([])
        for j in range(math.ceil(random_num[i])):
            dag_list[i].append(j)
        generate_num += len(dag_list[i])

    if generate_num != args.n:
        if generate_num < args.n:
            for i in range(args.n - generate_num):
                index = rd.randrange(0, length, 1)
                dag_list[index].append(len(dag_list[index]))
        if generate_num > args.n:
            i = 0
            while i < generate_num - args.n:
                index = rd.randrange(0, length, 1)
                if len(dag_list[index]) <= 1:
                    continue
                else:
                    del dag_list[index][-1]
                    i += 1

    dag_list_update = []
    pos = 1
    max_pos = 0
    for i in range(length):
        dag_list_update.append(list(range(dag_num, dag_num + len(dag_list[i]))))
        dag_num += len(dag_list_update[i])
        pos = 1
        for j in dag_list_update[i]:
            position[j] = (3 * (i + 1), pos)
            pos += 5
        max_pos = pos if pos > max_pos else max_pos
        position['Start'] = (0, max_pos / 2)
        position['Exit'] = (3 * (length + 1), max_pos / 2)

    ############################################link#####################################################
    into_degree = [0] * args.n
    out_degree = [0] * args.n
    edges = []
    pred = 0

    for i in range(length - 1):
        sample_list = list(range(len(dag_list_update[i + 1])))
        for j in range(len(dag_list_update[i])):
            od = rd.randrange(1, args.max_out + 1, 1)
            od = len(dag_list_update[i + 1]) if len(dag_list_update[i + 1]) < od else od
            bridge = rd.sample(sample_list, od)
            for k in bridge:
                edges.append((dag_list_update[i][j], dag_list_update[i + 1][k]))
                into_degree[pred + len(dag_list_update[i]) + k] += 1
                out_degree[pred + j] += 1
        pred += len(dag_list_update[i])

    ######################################create start node and exit node################################
    for node, id in enumerate(into_degree):  # Add an entry node as a parent to all nodes without edges
        if id == 0:
            edges.append(('Start', node + 1))
            into_degree[node] += 1

    for node, od in enumerate(out_degree):  # Add exit nodes as children to all nodes without edges
        if od == 0:
            edges.append((node + 1, 'Exit'))
            out_degree[node] += 1

    #############################################plot###################################################
    return edges, into_degree, out_degree, position


def plot_DAG(edges, postion, filePath="outputs/DAG.png"):
    g1 = nx.DiGraph()
    g1.add_edges_from(edges)
    nx.draw_networkx(g1, arrows=True, pos=postion)
    plt.savefig(filePath, format="PNG")
    plt.show()
    return plt.clf


# def search_for_successors(node, edges):
#         '''
#         find successor node
#          :param node: the node id to be searched
#          :param edges: DAG edge information (Note that it is better to pass the value of the list (edges[:]) instead of the address of the list (edges)!!!)
#          :return: node's follow-up node id list
#         '''
#         map = {}
#         if node == 'Exit': return print("error, 'Exit' node do not have successors!")
#         for i in range(len(edges)):
#             if edges[i][0] in map.keys():
#                 map[edges[i][0]].append(edges[i][1])
#             else:
#                 map[edges[i][0]] = [edges[i][1]]
#         pred = map[node]
#         return pred
#
# def search_for_all_successors(node, edges):
#     save = node
#     node = [node]
#     for ele in node:
#         succ = search_for_successors(ele,edges)
#         if(len(succ)==1 and succ[0]=='Exit'):
#             break
#         for item in succ:
#             if item in node:
#                 continue
#             else:
#                 node.append(item)
#     node.remove(save)
#     return node
#
#
# def search_for_predecessor(node, edges):
#     '''
#     Find the predecessor node
#      :param node: the node id to be searched
#      :param edges: DAG edge information
#      :return: node's predecessor node id list
#     '''
#     map = {}
#     if node == 'Start': return print("error, 'Start' node do not have predecessor!")
#     for i in range(len(edges)):
#         if edges[i][1] in map.keys():
#             map[edges[i][1]].append(edges[i][0])
#         else:
#             map[edges[i][1]] = [edges[i][0]]
#     succ = map[node]
#     return succ
#
#
#
# def workflows_generator(mode='default', n=10, max_out=2, alpha=1, beta=1.0, t_unit=10, resource_unit=100):
#     '''
#     Randomly generate a DAG task and randomly assign its duration and (CPU, Memory) requirements
#      :param mode: DAG is generated by default parameters
#      :param n: number of tasks in the DAG
#      :para max_out: The maximum number of child nodes of a DAG node
#      :return: edges DAG edge information
#               duration DAG node duration
#               demand DAG node resource requirement quantity
#               position position in the drawing
#     '''
#     t = t_unit  # s   time unit
#     r = resource_unit  # resource unit
#     edges, in_degree, out_degree, position = DAGs_generate(mode, n, max_out, alpha, beta)
#     plot_DAG(edges,position)
#     duration = []
#     demand = []
#     # initialization duration
#     for i in range(len(in_degree)):
#         if rd.random() < args.prob:
#             # duration.append(random.uniform(t,3*t))
#             duration.append(rd.sample(range(0, 3 * t), 1)[0])
#         else:
#             # duration.append(random.uniform(5*t,10*t))
#             duration.append(rd.sample(range(5 * t, 10 * t), 1)[0])
#     # Initial resource requirements
#     for i in range(len(in_degree)):
#         if rd.random() < 0.5:
#             demand.append((rd.uniform(0.25 * r, 0.5 * r), rd.uniform(0.05 * r, 0.01 * r)))
#         else:
#             demand.append((rd.uniform(0.05 * r, 0.01 * r), rd.uniform(0.25 * r, 0.5 * r)))
#
#     return edges, duration, demand, position


if __name__ == "__main__":
    np.random.seed(8)
    rd.seed(8)
    num_nodes = 10
    mode = 'default'
    max_out = 2
    alpha = 1
    beta = 1.0
    edges, in_degree, out_degree, position = DAGs_generate(mode, num_nodes, max_out, alpha, beta)
    print("edges=", edges)
    print("in_degree=", in_degree)
    print("out_degree=", out_degree)
    print("position=", position)
    savePath = "outputs/DAG.png"
    plot_DAG(edges, position, savePath)

    # dag = rand.directed_erdos(nnodes, .5)
    # print("dag=", dag)
    # nxGraph = dag.to_nx()
    # savePath = "outputs/DAG.png"
    # plot_DAG(nxGraph,savePath,True)
    # edges, duration, demand, position = workflows_generator()
    # print(edges)
