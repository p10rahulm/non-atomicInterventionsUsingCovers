from typing import Tuple

import numpy as np, scipy as sc, random as rd


class CoveredGraph:
    def __new__(cls, *args, **kwargs):
        # print("1. Create a new instance of the graph.")
        return super().__new__(cls)

    def __init__(self, num_vertices=15, degree=3, calA_size=10,
                 possible_prob_choices=None, prob_choice_weights=None,
                 best_parent_prob=0.99,
                 num_interventions_in_calA=10, size_of_intervention_in_calA=3,
                 initialQValues=0.01, pi=0.05, epsilon=0.1):
        # Replacing mutable (list) defaults with values in the init function
        if prob_choice_weights is None:
            prob_choice_weights = [3, 3, 1]
        if possible_prob_choices is None:
            possible_prob_choices = [0.1, 0.9, 0]
        self.num_vertices = num_vertices
        self.degree = degree
        self.calA_size = calA_size
        self.possible_prob_choices = possible_prob_choices
        self.prob_choice_weights = prob_choice_weights
        self.best_parent_prob = best_parent_prob
        # print("Initialize the new instance of graph.")
        self.graph, self.numOfParentsPerVertex = self.getGraph
        self.best_parent_of_y = self.getBestParentOfY
        self.conditionalProbs = self.getConditionalProbs
        self.num_interventions_in_calA = num_interventions_in_calA
        self.size_of_intervention_in_calA = size_of_intervention_in_calA
        self.calA = self.get_cal_A
        self.unconditionalProbs = self.computeProbOfOne

    def __repr__(self) -> str:
        return f"{type(self).__name__}(degree={self.degree}, num_vertices={self.num_vertices}, " \
               f"\ngraph={self.graph}\n numOfParentsPerVertex={self.numOfParentsPerVertex}\n" \
               f"\nconditionalProbsOfOne={self.conditionalProbs}\n calA={self.calA}\n" \
               f"\nunconditionalProbs={self.unconditionalProbs}\n"

    @property
    def getGraph(self):
        my_graph = np.empty((self.num_vertices, self.degree))
        my_graph[:] = np.nan
        num_of_parents_per_vertex = np.zeros(self.num_vertices, dtype=int)
        for i in range(1, self.num_vertices):
            set_of_parents = sorted(list(set(rd.choices(range(i), k=self.degree))))
            num_parents_i = len(set_of_parents)
            num_of_parents_per_vertex[i] = num_parents_i
            my_graph[i][:num_parents_i] = set_of_parents

        return my_graph, num_of_parents_per_vertex

    @property
    def getBestParentOfY(self):
        # best_parent_of_y = rd.choice(range(2 ** self.numOfParentsPerVertex[-1]))
        # We can arbitrarily choose to set best parent as 0, the case where all parents are 0.
        best_parent_of_y = 0
        return best_parent_of_y

    @property
    def getConditionalProbs(self):
        max_degree = np.max(self.numOfParentsPerVertex)
        cond_probs = np.empty((self.num_vertices, 2 ** max_degree))
        cond_probs[:] = np.nan
        possible_prob_choices = self.possible_prob_choices
        choice_weights = self.prob_choice_weights
        for i in range(self.num_vertices):
            num_parents = self.numOfParentsPerVertex[i]
            num_parent_combinations = int(2 ** num_parents)
            choice_list = list(rd.choices(population=possible_prob_choices, weights=choice_weights,
                                          k=num_parent_combinations))
            cond_probs[i, :num_parent_combinations] = [0] * num_parent_combinations
            cond_probs[i][:num_parent_combinations] = choice_list
        cond_probs[-1, self.best_parent_of_y] = self.best_parent_prob
        return cond_probs

    @property
    # Say only  interventions are allowed
    def get_cal_A(self):
        num_interventions = self.num_interventions_in_calA
        print("num_interventions=", num_interventions)
        calA = []
        for i in range(num_interventions):
            chosen_nodes = rd.sample(range(self.num_vertices), self.size_of_intervention_in_calA)
            A = []
            for node in chosen_nodes:
                value = rd.choice(range(2))
                A.append((node, value))
            calA.append(A)
        return calA

    @property
    def computeProbOfOne(self):
        cond_probs = self.conditionalProbs
        num_vertices = self.num_vertices
        unconditional_probs = np.zeros(num_vertices)
        unconditional_probs[0] = cond_probs[0, 0]
        conditional_probs_given_parents = np.copy(cond_probs)
        given_graph = self.graph
        for i in range(1, num_vertices):
            parents_of_i = [int(elem) for elem in given_graph[i] if ~np.isnan(elem)]
            num_parents = len(parents_of_i)
            num_combos = 2 ** num_parents
            unconditional_prob_parents = unconditional_probs[parents_of_i]
            cond_probs_i = cond_probs[i, :num_combos]
            prob_of_parent_combinations = self.get_prob_of_parent_combinations(unconditional_prob_parents, num_combos)
            conditional_probs_given_parents[i, :num_combos] = prob_of_parent_combinations
            unconditional_probs[i] = cond_probs_i @ prob_of_parent_combinations
            print("\ni=", i, "parents", parents_of_i,
                  "\nselected_unconditional_probs=", unconditional_prob_parents, "\ncond_probs_i=", cond_probs_i,
                  "\nprob_of_parent_combinations=", prob_of_parent_combinations,
                  "\nunconditional_probs[i]=", unconditional_probs[i])

        return unconditional_probs

    @staticmethod
    def get_prob_of_parent_combinations(unconditional_prob_parents, num_combinations):
        prob1 = unconditional_prob_parents
        prob0 = 1 - unconditional_prob_parents
        prob_of_parent_combinations = np.zeros(num_combinations)
        for combination_index in range(num_combinations):
            choices_on_parents = [int(elem) for elem in list(bin(combination_index)[2:])]
            multiplier = 1
            for parent_i in range(len(choices_on_parents)):
                multiplier_for_parent = prob1[parent_i] if choices_on_parents[parent_i] == 1 else prob0[parent_i]
                multiplier *= multiplier_for_parent
            prob_of_parent_combinations[combination_index] = multiplier
        return prob_of_parent_combinations

        return 0


if __name__ == "__main__":
    graph = CoveredGraph(15, 3)
    print("graph=", graph)
