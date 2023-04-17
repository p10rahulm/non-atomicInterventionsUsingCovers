from experiment import Experiment
from causalBandit import CausalBandit
import time
import numpy as np
import random as rd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


class Simulation:
    def __new__(cls, *args, **kwargs):
        # print("1. Create a new instance of the graph.")
        return super().__new__(cls)

    def __init__(self, experiment_types=None,
                 experiment_variable="time",
                 exp_variable_choices=None,
                 experiments_per_data_point=100,
                 num_vertices=15, degree=3, cal_a_size=10,
                 possible_prob_choices=None, prob_choice_weights=None,
                 best_parent_prob=0.99,
                 num_interventions_in_cal_a=10, size_of_intervention_in_cal_a=3,
                 cal_a_interventions_in_first_k_nodes=10):
        if prob_choice_weights is None:
            prob_choice_weights = [6, 3, 1]
        if possible_prob_choices is None:
            possible_prob_choices = [0, 0.01, 0.05]
        if exp_variable_choices is None:
            exp_variable_choices = list(range(1000, 11000, 1000))
        if experiment_types is None:
            experiment_types = ["yabe", "covers"]
        self.experiment_types = experiment_types
        self.experiments_per_data_point = experiments_per_data_point
        self.experiment_variable = experiment_variable
        self.exp_variable_choices = exp_variable_choices

        self.num_vertices = num_vertices
        self.degree = degree
        self.cal_a_size = cal_a_size
        self.possible_prob_choices = possible_prob_choices
        self.prob_choice_weights = prob_choice_weights
        self.best_parent_prob = best_parent_prob
        self.num_interventions_in_cal_a = num_interventions_in_cal_a
        self.size_of_intervention_in_cal_a = size_of_intervention_in_cal_a
        self.cal_a_interventions_in_first_k_nodes = cal_a_interventions_in_first_k_nodes

        self.experiment_values = self.run_simulation
        Simulation.show_plot(self.experiment_values, exp_variable_choices,
                             "Plot for Simple Regret with Time Horizon", "Time Horizon T", "Simple Regret",
                             self.experiment_types, "outputs/")

    def __repr__(self) -> str:
        out_str = f"{type(self).__name__}(experiment_variable={self.experiment_variable}" \
                  f"\nexp_variable_choices={self.exp_variable_choices}\n" \
                  f"\nexperiment_types={self.experiment_types}\n" \
                  f"\nexperiment_values={self.experiment_values}"
        return out_str

    @property
    def run_simulation(self) -> object:
        sim_results = np.zeros((len(self.exp_variable_choices), len(self.experiment_types)))
        for var_index in tqdm(range(len(self.exp_variable_choices))):
            var = self.exp_variable_choices[var_index]
            for i in tqdm(range(self.experiments_per_data_point)):
                sim_causal_bandit = CausalBandit(self.num_vertices, self.degree, self.cal_a_size,
                                                 self.possible_prob_choices,
                                                 self.prob_choice_weights, self.best_parent_prob,
                                                 self.num_interventions_in_cal_a, self.size_of_intervention_in_cal_a,
                                                 self.cal_a_interventions_in_first_k_nodes)

                for experiment_type_index in range(len(self.experiment_types)):
                    experiment_type = self.experiment_types[experiment_type_index]
                    expt = Experiment(experiment_type=experiment_type, input_causal_bandit=sim_causal_bandit,
                                      num_total_interventions=var)
                    sim_results[var_index, experiment_type_index] += expt.expected_regret
        sim_results = sim_results / self.experiments_per_data_point
        return sim_results

    @staticmethod
    def show_plot(sim_results, x_labels, plot_title, x_axis_label, y_axis_label, legend, location):
        plt.figure(figsize=(6, 4))
        for i in range(sim_results.shape[1]):
            y_values = sim_results[:, i]
            x_values = x_labels
            ax = sns.lineplot(y=y_values, x=x_values, label=legend[i])
        plt.xlabel(x_axis_label, fontsize=14)
        plt.ylabel(y_axis_label, fontsize=14)
        plt.title(plot_title, fontsize=14)
        now = datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")
        plt.savefig(location + 'regretWithTime_' + date_time + '.png', format='png')
        plt.show()


if __name__ == "__main__":
    # record time taken
    start_time = time.time()
    rd.seed(7)
    np.random.seed(9)
    np.set_printoptions(precision=4, suppress=True)

    sim1 = Simulation(experiment_types=["yabe", "covers"],
                      experiment_variable="time",
                      exp_variable_choices=list(range(100, 1100, 100)) + list(range(1000, 11000, 1000)),
                      experiments_per_data_point=300,
                      num_vertices=24, degree=4, cal_a_size=10,
                      possible_prob_choices=[0, 0.01, 0.05], prob_choice_weights=[6, 3, 1],
                      best_parent_prob=0.99,
                      num_interventions_in_cal_a=10, size_of_intervention_in_cal_a=3,
                      cal_a_interventions_in_first_k_nodes=10)
    print("sim1=", sim1)

    print("time taken=", time.time() - start_time)
