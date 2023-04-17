import time
import numpy as np
import random as rd
import statistics as stats
from tqdm import tqdm

from causalBandit import CausalBandit


# noinspection PyTypeChecker
class Experiment:
    def __new__(cls, *args, **kwargs):
        # print("1. Create a new instance of the graph.")
        return super().__new__(cls)

    def __init__(self, experiment_type="yabe",
                 input_causal_bandit=CausalBandit(),
                 num_total_interventions=1000):
        self.experiment_type = experiment_type
        self.num_total_interventions = num_total_interventions
        self.causal_bandit = input_causal_bandit
        self.graph = self.causal_bandit.graph
        self.num_vertices = self.causal_bandit.num_vertices
        self.expected_y_for_cal_a = self.causal_bandit.expected_y_for_cal_a
        self.best_intervention_index = self.causal_bandit.best_intervention_index
        self.best_intervention_expected_reward = self.causal_bandit.best_intervention_expected_reward
        self.cal_a = self.causal_bandit.cal_a
        self.cal_a_size = len(self.cal_a)
        self.num_interventions_per_cal_a = self.num_total_interventions // self.cal_a_size
        self.list_of_num_interventions_per_cal_a = [self.num_interventions_per_cal_a] * self.cal_a_size

        if experiment_type == "direct_bandit":
            self.unconditional_pr1_for_cal_a = self.causal_bandit.unconditional_pr1_for_cal_a
            self.simulated_vertex_values = Experiment.simulate_vertices(self.unconditional_pr1_for_cal_a,
                                                                        self.list_of_num_interventions_per_cal_a,
                                                                        self.cal_a_size, self.num_vertices)
            self.simulated_y = np.array([elem[-1] for elem in self.simulated_vertex_values])
            self.best_simulated_intervention_index = np.argmax(self.simulated_y)
            self.expected_reward_of_chosen_intervention = self.expected_y_for_cal_a[
                self.best_simulated_intervention_index]
            self.expected_regret = self.best_intervention_expected_reward - self.expected_reward_of_chosen_intervention

        elif experiment_type == "yabe":
            self.conditional_probs = self.causal_bandit.conditional_probs
            self.conditional_pr_parent_occurrence = self.causal_bandit.conditional_pr_parent_occurrence
            self.simulated_conditional_pr1_given_parents = \
                Experiment.simulate_conditional_prob_given_parents(self.conditional_pr_parent_occurrence,
                                                                   self.conditional_probs,
                                                                   self.num_vertices, 2 ** self.causal_bandit.degree,
                                                                   self.cal_a, self.list_of_num_interventions_per_cal_a)
            self.simulated_expected_vertex_values, self.simulated_cond_pr_parent_occurrence = \
                CausalBandit.probs_of_1_for_cal_a(self.simulated_conditional_pr1_given_parents,
                                                  self.graph, self.cal_a, self.num_vertices)

            self.simulated_y = np.array([elem[-1] for elem in self.simulated_expected_vertex_values])
            self.best_simulated_intervention_index = np.argmax(self.simulated_y)
            self.expected_reward_of_chosen_intervention = self.expected_y_for_cal_a[
                self.best_simulated_intervention_index]
            self.expected_regret = self.best_intervention_expected_reward - self.expected_reward_of_chosen_intervention

        elif experiment_type == "covers":
            self.conditional_probs = self.causal_bandit.conditional_probs
            self.numOfParentsPerVertex = self.causal_bandit.numOfParentsPerVertex
            self.cal_a_interventions_in_first_k_nodes = self.causal_bandit.cal_a_interventions_in_first_k_nodes
            self.cal_i = Experiment.get_cover_set(self.graph, self.numOfParentsPerVertex,
                                                  self.cal_a_interventions_in_first_k_nodes)
            self.num_covers = len(self.cal_i)
            self.num_interventions_per_cal_i = [self.num_total_interventions // self.num_covers] * self.num_covers
            _, self.conditional_pr_parent_occurrence_cal_i = \
                CausalBandit.probs_of_1_for_cal_a(self.conditional_probs, self.graph, self.cal_i, self.num_vertices)

            self.simulated_conditional_pr1_given_parents = \
                Experiment.simulate_conditional_prob_given_parents(self.conditional_pr_parent_occurrence_cal_i,
                                                                   self.conditional_probs,
                                                                   self.num_vertices, 2 ** self.causal_bandit.degree,
                                                                   self.cal_i, self.num_interventions_per_cal_i)
            self.simulated_expected_vertex_values, _ = \
                CausalBandit.probs_of_1_for_cal_a(self.simulated_conditional_pr1_given_parents,
                                                  self.graph, self.cal_a, self.num_vertices)

            self.simulated_y = np.array([elem[-1] for elem in self.simulated_expected_vertex_values])
            self.best_simulated_intervention_index = np.argmax(self.simulated_y)
            self.expected_reward_of_chosen_intervention = self.expected_y_for_cal_a[
                self.best_simulated_intervention_index]
            self.expected_regret = self.best_intervention_expected_reward - self.expected_reward_of_chosen_intervention

    def __repr__(self) -> str:
        common_string = f"{type(self).__name__}(num_total_interventions={self.num_total_interventions}" \
                        f"\ncal_a={self.cal_a}\n" \
                        f"\ncal_a_size={self.cal_a_size}" \
                        f"\nnum_interventions_per_cal_a={self.num_interventions_per_cal_a}" \
                        f"\nlist_of_num_interventions_per_cal_a={self.list_of_num_interventions_per_cal_a}"

        if self.experiment_type == "direct_bandit":
            direct_bandit_str = f"\nunconditional_pr1_for_cal_a={self.unconditional_pr1_for_cal_a}" \
                                f"\nsimulated_vertex_values={self.simulated_vertex_values}"
            output_str = f"\nsimulated_y={self.simulated_y}" \
                         f"\nbest_simulated_intervention_index={self.best_simulated_intervention_index}" \
                         f"\nexpected_reward_of_chosen_intervention={self.expected_reward_of_chosen_intervention}" \
                         f"\nexpected_regret={self.expected_regret}"
            return common_string + direct_bandit_str + output_str

        elif self.experiment_type == "yabe":
            yabe_str = f"\nsimulated_cond_pr_parent_occurrence={self.simulated_cond_pr_parent_occurrence}" \
                       f"\nsimulated_cond_pr_parent_occurrence[0]={self.simulated_cond_pr_parent_occurrence[0]}" \
                       f"\nconditional_probs={self.conditional_probs}" \
                       f"\nsimulated_expected_vertex_values={self.simulated_expected_vertex_values}"
            # f"\nsimulated_conditional_pr1_given_parents={self.simulated_conditional_pr1_given_parents}" \

            output_str = f"\nsimulated_y={self.simulated_y}" \
                         f"\nbest_simulated_intervention_index={self.best_simulated_intervention_index}" \
                         f"\nexpected_reward_of_chosen_intervention={self.expected_reward_of_chosen_intervention}" \
                         f"\nexpected_regret={self.expected_regret}"

            return common_string + yabe_str + output_str

        elif self.experiment_type == "covers":
            covers_str = f"\nnumOfParentsPerVertex={self.numOfParentsPerVertex}" \
                         f"\ncal_i={self.cal_i}\nnum_covers={self.num_covers}" \
                         f"\nnum_interventions_per_cal_i={self.num_interventions_per_cal_i}" \
                         f"\nsimulated_conditional_pr1_given_parents={self.simulated_conditional_pr1_given_parents}" \
                         f"\nsimulated_expected_vertex_values={self.simulated_expected_vertex_values}"
            output_str = f"\nsimulated_y={self.simulated_y}" \
                         f"\nbest_simulated_intervention_index={self.best_simulated_intervention_index}" \
                         f"\nexpected_reward_of_chosen_intervention={self.expected_reward_of_chosen_intervention}" \
                         f"\nexpected_regret={self.expected_regret}"

            return common_string + covers_str + output_str
        return common_string

    @staticmethod
    def simulate_vertices(unconditional_pr1_for_cal_a, list_of_num_interventions_per_cal_a, cal_a_size, num_vertices):
        # pr1 = unconditional_pr1_for_cal_a
        # list_of_num_interventions_per_cal_a = self.list_of_num_interventions_per_cal_a
        # cal_a_size = self.cal_a_size
        # num_vertices = self.num_vertices
        pr1_in_cal_a_experiment = []
        for i in range(cal_a_size):
            num1s = np.zeros(num_vertices)
            num_trials_for_this_intervention = list_of_num_interventions_per_cal_a[i]
            num_trials = np.ones(num_vertices) * num_trials_for_this_intervention
            pr1_in_intervention = unconditional_pr1_for_cal_a[i]

            for j in range(len(pr1_in_intervention)):
                pi = pr1_in_intervention[j]
                num1s[j] = np.random.binomial(n=num_trials_for_this_intervention, p=pi)

            prob_1_for_intervention = np.divide(num1s, num_trials,
                                                out=np.zeros(num1s.shape, dtype=float), where=num_trials != 0)
            # prob_1_for_intervention = num1s / num_trials
            pr1_in_cal_a_experiment.append(prob_1_for_intervention)
        return pr1_in_cal_a_experiment

    @staticmethod
    def simulate_conditional_prob_given_parents(conditional_pr_parent_occurrence, conditional_probs, num_vertices,
                                                num_parent_combinations, cal_a, list_of_num_interventions_per_cal_a):
        cal_a_size = len(cal_a)
        num_trials = np.zeros((num_vertices, num_parent_combinations), dtype=int)
        num_1s = np.zeros((num_vertices, num_parent_combinations), dtype=int)
        for i in range(cal_a_size):
            interventions = dict(cal_a[i])
            # print("interventions=", interventions)
            cond_pr_parent_occurrence = conditional_pr_parent_occurrence[i]
            num_trials_addition_table = np.zeros(cond_pr_parent_occurrence.shape, dtype=int)
            num_interventions = list_of_num_interventions_per_cal_a[i]
            num_1s_addition_table = np.zeros(cond_pr_parent_occurrence.shape, dtype=int)
            for row in range(cond_pr_parent_occurrence.shape[0]):
                if row not in interventions:
                    for col in range(cond_pr_parent_occurrence.shape[1]):
                        if ~np.isnan(cond_pr_parent_occurrence[row, col]):
                            num_trials_addition_table[row, col] = num_interventions * \
                              cond_pr_parent_occurrence[row, col] if row != 0 and col != 0 else num_interventions
                            num_1s_addition_table[row, col] = np.random.binomial(n=num_trials_addition_table[row, col],
                                                                                 p=conditional_probs[row, col])
            num_1s += num_1s_addition_table
            num_trials += num_trials_addition_table
            # print("\n\nnum_interventions=",num_interventions,"interventions=", interventions,
            #       "cond_pr_parent_occurrence=", cond_pr_parent_occurrence,
            #       "num_1s_addition_table=", num_1s_addition_table, "num_1s=", num_1s,
            #       "num_trials_addition_table=", num_trials_addition_table, "num_trials=", num_trials)
        # print("cal_a_size=",cal_a_size,"num_1s=",num_1s,"num_trials=",num_trials)
        with np.errstate(divide='ignore', invalid='ignore'):
            conditional_pr1_given_parents = np.divide(num_1s, num_trials,
                                                      out=np.zeros(num_1s.shape, dtype=float), where=num_trials != 0)
            conditional_pr1_given_parents[np.isnan(conditional_probs)] = np.nan
        return conditional_pr1_given_parents

    @staticmethod
    # importing Ayush's method
    def get_cover_set(graph, numOfParentsPerVertex, cal_a_interventions_in_first_k_nodes):
        num_vertices_for_cover = len(numOfParentsPerVertex)-1
        # num_vertices_for_cover = cal_a_interventions_in_first_k_nodes
        # d = max([len(vertex) for vertex in graph])  # max degree
        max_degree = max(numOfParentsPerVertex)  # max degree
        total_alphas_to_cover = 0
        for index in range(num_vertices_for_cover):
            if numOfParentsPerVertex[index] != 0:
                total_alphas_to_cover += 2 ** numOfParentsPerVertex[index]

        # initialize cal_i with empty intervention
        cal_i = [[]]
        alphas_covered = 0
        mark_covered_alphas = np.ndarray((num_vertices_for_cover, 2 ** max_degree))
        mark_covered_alphas.fill(0)
        p_of_intervening = max_degree / (max_degree + 1)
        # p_of_intervening = 0.5
        # print("p_of_intervening=", p_of_intervening)
        while alphas_covered < total_alphas_to_cover:
            intervention_set = {}
            nodes_intervened = set()
            something_new_covered = False
            for index in range(num_vertices_for_cover):
                if np.random.binomial(1, p_of_intervening) == 1:
                    intervention_set[index] = np.random.binomial(1, 0.5)
                    nodes_intervened.add(index)
            for index in range(num_vertices_for_cover):
                parents_of_i = [int(elem) for elem in graph[index] if ~np.isnan(elem)]
                if numOfParentsPerVertex[index] != 0 and \
                        index not in nodes_intervened and \
                        all(y in nodes_intervened for y in parents_of_i):
                    pa_val = np.array([intervention_set[pa] for pa in parents_of_i])
                    pa_index = int(pa_val.dot(1 << np.arange(pa_val.size)[::-1]))
                    if mark_covered_alphas[index, pa_index] == 0:
                        mark_covered_alphas[index, pa_index] = 1
                        alphas_covered += 1
                        something_new_covered = True
            if something_new_covered:  # only add to covering set if a new alpha is covered by intervention_set
                cal_i.append(list(intervention_set.items()))
        return cal_i


if __name__ == "__main__":
    # record time taken
    start_time = time.time()
    rd.seed(7)
    np.random.seed(9)
    np.set_printoptions(precision=4, suppress=True)

    causal_bandit = CausalBandit(num_vertices=15, degree=2, cal_a_size=10,
                                 possible_prob_choices=[0.1, 0.9, 0], prob_choice_weights=[3, 3, 1],
                                 best_parent_prob=0.99,
                                 num_interventions_in_cal_a=10, size_of_intervention_in_cal_a=3,
                                 cal_a_interventions_in_first_k_nodes=10)

    expt1 = Experiment(experiment_type="direct_bandit", input_causal_bandit=causal_bandit, num_total_interventions=500)
    print("expt1=", expt1)
    expt2 = Experiment(experiment_type="yabe", input_causal_bandit=causal_bandit, num_total_interventions=250)
    print("\n\n\nexpt2=", expt2)
    expt3 = Experiment(experiment_type="covers", input_causal_bandit=causal_bandit, num_total_interventions=250)
    print("\n\n\nexpt3=", expt3)
    # Run a simulation with multiple experiments
    regret_yabe, regret_covers = [], []
    num_simulations = 100
    for i in tqdm(range(num_simulations)):
        causal_bandit = CausalBandit(num_vertices=15, degree=3, cal_a_size=10,
                                     possible_prob_choices=[0, 0.01, 0.05], prob_choice_weights=[6, 3, 1],
                                     best_parent_prob=0.99,
                                     num_interventions_in_cal_a=10, size_of_intervention_in_cal_a=3,
                                     cal_a_interventions_in_first_k_nodes=10)
        expt1 = Experiment(experiment_type="yabe", input_causal_bandit=causal_bandit, num_total_interventions=500)
        expt2 = Experiment(experiment_type="covers", input_causal_bandit=causal_bandit, num_total_interventions=500)
        regret_yabe.append(expt1.expected_regret)
        regret_covers.append(expt2.expected_regret)

    average_regret_yabe = stats.fmean(regret_yabe)
    average_regret_covers = stats.fmean(regret_covers)
    print(f"\naverage_regret_yabe={average_regret_yabe}\naverage_regret_covers={average_regret_covers}")

    print("time taken=", time.time() - start_time)
