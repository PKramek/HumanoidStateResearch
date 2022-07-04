from typing import List

import numpy as np
from matplotlib import pyplot as plt


def get_reward_modifier(fi_xt: float, fi_xt_plus_1: float, gamma=0.99):
    return -fi_xt + gamma * fi_xt_plus_1


def print_fi_x_summary(fi_x: callable, optimal_state=None, worse_state=None):
    if optimal_state is None:
        optimal_state = [1.4]
    if worse_state is None:
        worse_state = [1.4 - 0.008]

    fi_optimal = fi_x(optimal_state)
    fi_worse = fi_x(worse_state)

    from_worse_to_optimal_reward_modifier = get_reward_modifier(fi_xt=fi_worse, fi_xt_plus_1=fi_optimal)
    from_optimal_to_worse_reward_modifier = get_reward_modifier(fi_xt=fi_optimal, fi_xt_plus_1=fi_worse)
    from_optimal_to_optimal_reward_modifier = get_reward_modifier(fi_xt=fi_optimal, fi_xt_plus_1=fi_optimal)

    print(f"Fi value at optimal state: {fi_optimal:.4f}")
    print(f"Fi value at worse state: {fi_worse:.4f}")
    print(f"Reward modifier when passing from worse to optimal state: {from_worse_to_optimal_reward_modifier}")
    print(f"Reward modifier when passing from optimal to worse state: {from_optimal_to_worse_reward_modifier}")
    print(f"Reward modifier when passing from optimal to optimal state: {from_optimal_to_optimal_reward_modifier}")


def get_plot_file_name(env_name: str, index: int) -> str:
    return f"{env_name}_state_vector_at_index_{index}"


def get_plot_title(env_name: str, index: int) -> str:
    return f"{env_name} state vector at index {index}"


def get_abs_path_to_file(abs_path: str, filename: str):
    if abs_path[-1] == '/':
        abs_path = abs_path[:-1]

    return f"{abs_path}/{filename}"


def plot_and_save_values(env_name: str, index: int, path: str, values: List[float]):
    plot_title = get_plot_title(env_name, index)
    filename = get_plot_file_name(env_name, index)
    abs_path_to_file = get_abs_path_to_file(path, filename)

    plt.plot(values)
    plt.xlabel("Steps")
    plt.ylabel(f"Value at index {index}")
    plt.grid()
    plt.title(plot_title)
    plt.savefig(abs_path_to_file)


def plot_fi_x(x: List, y: List, title: str):
    plt.plot(x, y)
    plt.xlabel("x[0]")
    plt.ylabel("fi(x)")
    plt.grid()
    plt.title(title)
    plt.show()


def normal_dist_density(x: float, mean: float, sd: float):
    prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density


def calculate_shaped_reward(reward: float, last_fi_value: float, fi_value: float, gamma: float):
    return reward - last_fi_value + fi_value * gamma
