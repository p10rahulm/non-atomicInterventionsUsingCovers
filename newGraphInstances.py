import time
import numpy as np, scipy as sc, random as rd


class CoveredGraph:
    def __new__(cls, *args, **kwargs):
        # print("1. Create a new instance of the graph.")
        return super().__new__(cls)

    def __init__(self, num_vertices=15, degree=3, cal_a_size=10,
                 possible_prob_choices=None, prob_choice_weights=None,
                 best_parent_prob=0.99,
                 num_interventions_in_cal_a=10, size_of_intervention_in_cal_a=3,
                 cal_a_interventions_in_first_k_nodes=10,
                 initialQValues=0.01, pi=0.05, epsilon=0.1):
        # Replacing mutable (list) defaults with values in the init function
        if prob_choice_weights is None:
            prob_choice_weights = [3, 3, 1]
        if possible_prob_choices is None:
            possible_prob_choices = [0.1, 0.9, 0]
        self.num_vertices = num_vertices
        self.degree = degree
        self.cal_a_size = cal_a_size
        self.possible_prob_choices = possible_prob_choices
        self.prob_choice_weights = prob_choice_weights
        self.best_parent_prob = best_parent_prob
        # print("Initialize the new instance of graph.")
        self.graph, self.numOfParentsPerVertex = self.getGraph
        self.best_parent_of_y = self.getBestParentOfY
        self.conditionalProbs = self.getConditionalProbs
        self.num_interventions_in_cal_a = num_interventions_in_cal_a
        self.size_of_intervention_in_cal_a = size_of_intervention_in_cal_a
        self.cal_a_interventions_in_first_k_nodes = cal_a_interventions_in_first_k_nodes
        self.cal_a = self.get_cal_a
        self.unconditional_pr1_do_nothing, self.conditional_pr1_given_pa_do_nothing = self.computeProbOfOne
        self.unconditional_pr1_for_cal_a, self.conditional_pr1_given_pa_cal_a = self.probs_of_1_for_cal_a
        self.expected_y_for_cal_a = self.get_expected_y_vals_for_cal_a
        self.best_intervention_index = np.argmax(self.expected_y_for_cal_a)
        self.best_intervention_expected_reward = self.expected_y_for_cal_a[self.best_intervention_index]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(degree={self.degree}, num_vertices={self.num_vertices}, " \
               f"\ngraph={self.graph}\nnumOfParentsPerVertex={self.numOfParentsPerVertex}\n" \
               f"\nconditionalProbsOfOne={self.conditionalProbs}\ncal_a={self.cal_a}" \
               f"\nunconditional_pr1_do_nothing={self.unconditional_pr1_do_nothing}" \
               f"\nconditional_pr1_given_pa_do_nothing={self.conditional_pr1_given_pa_do_nothing}" \
               f"\nunconditional_pr1_for_cal_a={self.unconditional_pr1_for_cal_a}" \
               f"\nconditional_pr1_given_pa_cal_a={self.conditional_pr1_given_pa_cal_a}" \
               f"\nexpected_y_for_cal_a={self.expected_y_for_cal_a}" \
               f"\nself.best_intervention_index={self.best_intervention_index}" \
               f"\nself.best_intervention_expected_reward={self.best_intervention_expected_reward}"

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
    def get_cal_a(self):
        num_interventions = self.num_interventions_in_cal_a
        vertices_to_choose_from = self.cal_a_interventions_in_first_k_nodes
        cal_a = []
        # Add the do-nothing empty intervention
        cal_a.append([])
        for i in range(num_interventions):
            chosen_nodes = sorted(rd.sample(range(vertices_to_choose_from), self.size_of_intervention_in_cal_a))
            intervention_i = []
            for node in chosen_nodes:
                value = rd.choice(range(2))
                intervention_i.append((node, value))
            cal_a.append(intervention_i)
        return cal_a

    @property
    def computeProbOfOne(self):
        cond_probs = self.conditionalProbs
        num_vertices = self.num_vertices
        unconditional_probs_of_one = np.zeros(num_vertices)
        unconditional_probs_of_one[0] = cond_probs[0, 0]
        conditional_probs_given_parents = np.copy(cond_probs)
        given_graph = self.graph
        for i in range(1, num_vertices):
            parents_of_i = [int(elem) for elem in given_graph[i] if ~np.isnan(elem)]
            num_parents = len(parents_of_i)
            num_combos = 2 ** num_parents
            unconditional_prob_parents = unconditional_probs_of_one[parents_of_i]
            cond_probs_i = cond_probs[i, :num_combos]
            prob_of_parent_combinations = self.get_prob_of_parent_combinations(unconditional_prob_parents, num_combos)
            conditional_probs_given_parents[i, :num_combos] = prob_of_parent_combinations
            unconditional_probs_of_one[i] = cond_probs_i @ prob_of_parent_combinations
        return unconditional_probs_of_one, conditional_probs_given_parents

    @staticmethod
    def get_prob_of_parent_combinations(unconditional_prob_parents, num_combinations):
        prob1 = unconditional_prob_parents
        prob0 = 1 - unconditional_prob_parents
        prob_of_parent_combinations = np.zeros(num_combinations)
        num_digits = len(prob1)
        for combination_index in range(num_combinations):
            # Get number of digits
            binary_string = bin(combination_index)[2:].zfill(num_digits)
            choices_on_parents = [int(elem) for elem in list(binary_string)]
            multiplier = 1
            for index_parent in range(len(choices_on_parents)):
                multiplier_pa = prob1[index_parent] if choices_on_parents[index_parent] == 1 else prob0[index_parent]
                multiplier *= multiplier_pa
            prob_of_parent_combinations[combination_index] = multiplier
        return prob_of_parent_combinations

    @property
    def probs_of_1_for_cal_a(self):
        cal_a = self.cal_a
        cond_probs = self.conditionalProbs
        num_vertices = self.num_vertices
        given_graph = self.graph
        unconditional_pr1_for_cal_a = []
        conditional_pr1_given_pa_cal_a = []
        for intervention in cal_a:
            dict_of_intervention = dict(intervention)
            unconditional_probs_of_one = np.zeros(num_vertices)
            zero_vertex_intervention = dict_of_intervention[0] if 0 in dict_of_intervention else cond_probs[0, 0]
            unconditional_probs_of_one[0] = zero_vertex_intervention
            conditional_probs_given_parents = np.copy(cond_probs)
            conditional_probs_given_parents[0, 0] = zero_vertex_intervention
            for i in range(1, num_vertices):
                parents_of_i = [int(elem) for elem in given_graph[i] if ~np.isnan(elem)]
                num_parents = len(parents_of_i)
                num_combos = 2 ** num_parents
                unconditional_prob_pa = unconditional_probs_of_one[parents_of_i]
                cond_probs_i = cond_probs[i, :num_combos]
                prob_of_parent_combinations = self.get_prob_of_parent_combinations(unconditional_prob_pa, num_combos) \
                    if i not in dict_of_intervention else np.ones(num_combos) * dict_of_intervention[i]
                conditional_probs_given_parents[i, :num_combos] = prob_of_parent_combinations
                computed_prob = cond_probs_i @ prob_of_parent_combinations
                unconditional_probs_of_one[i] = dict_of_intervention[i] if i in dict_of_intervention else computed_prob
            unconditional_pr1_for_cal_a.append(unconditional_probs_of_one)
            conditional_pr1_given_pa_cal_a.append(conditional_probs_given_parents)

        return unconditional_pr1_for_cal_a, conditional_pr1_given_pa_cal_a

    @property
    def get_expected_y_vals_for_cal_a(self):
        uncond_pr1 = self.unconditional_pr1_for_cal_a
        uncond_y = np.array([elem[-1] for elem in uncond_pr1])
        return uncond_y


class DirectBanditExperiment:
    def __new__(cls, *args, **kwargs):
        # print("1. Create a new instance of the graph.")
        return super().__new__(cls)

    def __init__(self, input_graph=CoveredGraph(), num_total_interventions=1000):
        self.num_total_interventions = num_total_interventions
        self.graph = input_graph
        self.num_vertices = self.graph.num_vertices
        self.expected_y_for_cal_a = self.graph.expected_y_for_cal_a
        self.best_intervention_index = self.graph.best_intervention_index
        self.best_intervention_expected_reward = self.graph.best_intervention_expected_reward
        self.cal_a = self.graph.cal_a
        self.cal_a_size = len(self.cal_a)
        self.num_interventions_per_cal_a = self.num_total_interventions // self.cal_a_size
        self.interventions_per_cal_a = [self.num_interventions_per_cal_a] * self.cal_a_size
        self.prob_of_1 = self.graph.unconditional_pr1_for_cal_a
        self.conditional_pr1_given_pa_cal_a = self.graph.conditional_pr1_given_pa_cal_a
        self.simulated_vertex_values = self.simulate_vertices
        self.simulated_y = np.array([elem[-1] for elem in self.simulated_vertex_values])
        self.best_simulated_intervention_index = np.argmax(self.simulated_y)
        self.expected_reward_of_chosen_intervention = self.expected_y_for_cal_a[self.best_simulated_intervention_index]
        self.expected_regret = self.best_intervention_expected_reward - self.expected_reward_of_chosen_intervention

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_total_interventions={self.num_total_interventions}" \
               f"\ngraph={self.graph}\ncal_a={self.cal_a}\n" \
               f"\ncal_a_size={self.cal_a_size}\nnum_interventions_per_cal_a={self.num_interventions_per_cal_a}" \
               f"\ninterventions_per_cal_a={self.interventions_per_cal_a}" \
               f"\nprob_of_1={self.prob_of_1}" \
               f"\nsimulated_vertex_values={self.simulated_vertex_values}" \
               f"\nsimulated_y={self.simulated_y}" \
               f"\nbest_simulated_intervention_index={self.best_simulated_intervention_index}" \
               f"\nexpected_reward_of_chosen_intervention={self.expected_reward_of_chosen_intervention}" \
               f"\nexpected_regret={self.expected_regret}"

    @property
    def simulate_vertices(self):
        pr1 = self.prob_of_1
        interventions_per_cal_a = self.interventions_per_cal_a
        cal_a_size = self.cal_a_size
        num_vertices = self.num_vertices
        pr1_in_cal_a_experiment = []
        for i in range(cal_a_size):
            num1s = np.zeros(num_vertices)
            num_trials_for_this_intervention = interventions_per_cal_a[i]
            num_trials = np.ones(num_vertices) * num_trials_for_this_intervention
            pr1_in_intervention = pr1[i]

            for j in range(len(pr1_in_intervention)):
                pi = pr1_in_intervention[j]
                num1s[j] = np.random.binomial(n=num_trials_for_this_intervention, p=pi)
            prob_1_for_intervention = num1s / num_trials
            pr1_in_cal_a_experiment.append(prob_1_for_intervention)
        return pr1_in_cal_a_experiment

class YabeExperiment:
    def __new__(cls, *args, **kwargs):
        # print("1. Create a new instance of the graph.")
        return super().__new__(cls)

    def __init__(self, input_graph=CoveredGraph(), num_total_interventions=1000):
        self.num_total_interventions = num_total_interventions
        self.graph = input_graph
        self.num_vertices = self.graph.num_vertices
        self.expected_y_for_cal_a = self.graph.expected_y_for_cal_a
        self.best_intervention_index = self.graph.best_intervention_index
        self.best_intervention_expected_reward = self.graph.best_intervention_expected_reward
        self.cal_a = self.graph.cal_a
        self.cal_a_size = len(self.cal_a)
        self.num_interventions_per_cal_a = self.num_total_interventions // self.cal_a_size
        self.interventions_per_cal_a = [self.num_interventions_per_cal_a] * self.cal_a_size
        self.prob_of_1 = self.graph.unconditional_pr1_for_cal_a
        self.conditional_pr1_given_pa_cal_a = self.graph.conditional_pr1_given_pa_cal_a
        self.simulated_vertex_values = self.simulate_vertices
        self.simulated_y = np.array([elem[-1] for elem in self.simulated_vertex_values])
        self.best_simulated_intervention_index = np.argmax(self.simulated_y)
        self.expected_reward_of_chosen_intervention = self.expected_y_for_cal_a[self.best_simulated_intervention_index]
        self.expected_regret = self.best_intervention_expected_reward - self.expected_reward_of_chosen_intervention

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_total_interventions={self.num_total_interventions}" \
               f"\ngraph={self.graph}\ncal_a={self.cal_a}\n" \
               f"\ncal_a_size={self.cal_a_size}\nnum_interventions_per_cal_a={self.num_interventions_per_cal_a}" \
               f"\ninterventions_per_cal_a={self.interventions_per_cal_a}" \
               f"\nprob_of_1={self.prob_of_1}" \
               f"\nsimulated_vertex_values={self.simulated_vertex_values}" \
               f"\nsimulated_y={self.simulated_y}" \
               f"\nbest_simulated_intervention_index={self.best_simulated_intervention_index}" \
               f"\nexpected_reward_of_chosen_intervention={self.expected_reward_of_chosen_intervention}" \
               f"\nexpected_regret={self.expected_regret}"

    @property
    def simulate_vertices(self):
        pr1 = self.prob_of_1
        interventions_per_cal_a = self.interventions_per_cal_a
        cal_a_size = self.cal_a_size
        num_vertices = self.num_vertices
        pr1_in_cal_a_experiment = []
        for i in range(cal_a_size):
            num1s = np.zeros(num_vertices)
            num_trials_for_this_intervention = interventions_per_cal_a[i]
            num_trials = np.ones(num_vertices) * num_trials_for_this_intervention
            pr1_in_intervention = pr1[i]

            for j in range(len(pr1_in_intervention)):
                pi = pr1_in_intervention[j]
                num1s[j] = np.random.binomial(n=num_trials_for_this_intervention, p=pi)
            prob_1_for_intervention = num1s / num_trials
            pr1_in_cal_a_experiment.append(prob_1_for_intervention)
        return pr1_in_cal_a_experiment




if __name__ == "__main__":
    # record time taken
    start_time = time.time()
    rd.seed(7)
    np.random.seed(9)
    np.set_printoptions(precision=4, suppress=True)

    graph = CoveredGraph(15, 3)
    print("graph=", graph)
    expt = DirectBanditExperiment()
    print("expt=", expt)

    print("time taken=", time.time() - start_time)
