import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

# Description of Humanoid-v1 state vector, it supposedly was not changed
# https://github.com/openai/gym/wiki/Humanoid-V1
from util import normal_dist_density, plot_rewards_prim_from_beginning_to_fall, \
    plot_rewards_prim_from_fall_to_beginning


def for_seminary(x: np.ndarray):
    index = 0
    return 0.0 if 1.3 < x[index] < 1.5 else -np.power((1.4 - x[index]) * 100, 2)


def for_seminary_correct_results(x: np.ndarray):
    index = 0
    return (10 * normal_dist_density(x[index], 1.4, 0.05)) + 500


def alive_reward(x: np.ndarray):
    return 500.0


def alive_penalty(x: np.ndarray):
    return -500.0


def t_student_penalty_df_0_2_scale_0_0_5(state: np.ndarray, optimal_point: float = 1.4):
    index = 0
    df = 0.2
    scale = 0.05
    loc = optimal_point

    return t.pdf(state[index], df=df, scale=scale, loc=loc)


def normal_from_seminary(state: np.ndarray, middle_of_normal_dist: float = 1.4):
    index = 0
    return 10 * normal_dist_density(state[index], middle_of_normal_dist, 0.05) + 500


def normal_from_seminary_bigger_diff(state: np.ndarray, middle_of_normal_dist: float = 1.4):
    index = 0
    return 15 * normal_dist_density(state[index], middle_of_normal_dist, 0.05) + 500


parser = argparse.ArgumentParser(description='Display information about state vector of gym environment')

parser.add_argument('--env', type=str, help='Name of gym environment', required=True)
parser.add_argument('--steps', type=int, help='Number of steps to perform in the environment', default=1000)
parser.add_argument('--plot-index', type=int, help='Index of value in observation vector to create plot of', default=1)
parser.add_argument('--plot-path', type=str, help='Path to directory, where created plots will be created')

if __name__ == '__main__':
    used_fi = normal_from_seminary
    base_reward = 0.0
    step_size = 0.01
    gamma = 0.99

    test_input = np.arange(1.2, 1.6, step_size)
    output = np.zeros_like(test_input)
    degrees_of_freedom = 0.2
    middle_of_distribution = 1.4
    scale = 0.05

    for i in range(len(test_input)):
        output[i] = used_fi([test_input[i]])

    plt.plot(test_input, output)
    plt.title(f"{used_fi.__name__}")
    plt.show()

    plot_rewards_prim_from_beginning_to_fall(base_reward=base_reward, step_size=step_size,
                                             fi=used_fi, gamma=gamma)
    plot_rewards_prim_from_fall_to_beginning(base_reward=base_reward, step_size=step_size,
                                             fi=used_fi, gamma=gamma)

    functions_and_titles = [
        (used_fi, f"{used_fi.__name__}"),

    ]

