from experiment import Experiment
from causalBandit import CausalBandit
import time
import numpy as np
import random as rd
import statistics as stats
from tqdm import tqdm


class Simulation:
    def __new__(cls, *args, **kwargs):
        # print("1. Create a new instance of the graph.")
        return super().__new__(cls)

    def __init__(self, experiment_type=["yabe", "covers"],
                 experiment_variable="time",
                 experiment_choices=list(range(1000, 11000, 1000))):
        self.experiment_type = experiment_type
        self.num_total_interventions = num_total_interventions


if __name__ == "__main__":
    # record time taken
    start_time = time.time()
    rd.seed(7)
    np.random.seed(9)
    np.set_printoptions(precision=4, suppress=True)

    causal_bandit = CausalBandit(num_vertices=15, degree=3, cal_a_size=10,
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
