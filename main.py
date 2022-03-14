import argparse
import timeit
from typing import List

import matplotlib.pyplot as plt
import numpy as np


# According to this thread: https://github.com/openai/gym/issues/585
# first 3 elements of the state vector are (X,Y,Z) coordinates

# Description of Humanoid-v1 state vector, it supposedly was not changed
# https://github.com/openai/gym/wiki/Humanoid-V1

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


def normal_dist_density(x: float, mean: float, sd: float):
    prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density


def fi_one_over_x(x: np.ndarray) -> float:
    return min(abs(1.0 / (1.4 - x[0])), 200)


def fi_one_over_sqrt(x: np.ndarray) -> float:
    diff = 1.4 - x[0]
    return min(1 / np.sqrt(diff * diff), 200)


def fi_normal_density(x: np.ndarray) -> float:
    return 1000 * normal_dist_density(x[0], 1.4, 0.05)


def plot_fi_x(x: List, y: List, title: str):
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("fi(x)")
    plt.grid()
    plt.title(title)
    plt.show()


parser = argparse.ArgumentParser(description='Display information about state vector of gym environment')
parser.add_argument('--env', type=str, help='Name of gym environment', required=True)
parser.add_argument('--steps', type=int, help='Number of steps to perform in the environment', default=1000)
parser.add_argument('--plot-index', type=int, help='Index of value in observation vector to create plot of', default=1)
parser.add_argument('--plot-path', type=str, help='Path to directory, where created plots will be created')

if __name__ == '__main__':
    test_values = np.arange(1.2, 1.6, 0.0001)
    test_values_as_vectors = list(map(lambda x: [x], test_values))

    functions_and_titles = [
        (fi_one_over_x, "fi(x) = min(1/abs(1.4-x), 200)"),
        (fi_one_over_sqrt, "fi(x) = min(1/sqrt(pow((1.4-x) , 2)), 200)"),
        (fi_normal_density, "fi(x) = normal(x, 1.4, 0.05) * 1000"),
        (fi_sum_of_normal_densities, "fi(x) = normal(x, 1.4, 0.05) + normal(x, 1.4, 0.001) * 1000")
    ]

    for function, plot_title in functions_and_titles:
        test_fi_x = np.array(list(map(function, test_values_as_vectors)))
        plot_fi_x(test_values, test_fi_x, plot_title)

    repetitions = 1000

    fi_one_over_x_result = timeit.timeit('list(map(fi_one_over_x, test_values_as_vectors))', number=repetitions,
                                         setup="from __main__ import fi_one_over_x, test_values_as_vectors")

    fi_one_over_sqrt_result = timeit.timeit('list(map(fi_one_over_sqrt, test_values_as_vectors))', number=repetitions,
                                            setup="from __main__ import fi_one_over_sqrt, test_values_as_vectors")
    fi_normal_density_result = timeit.timeit('list(map(fi_normal_density, test_values_as_vectors))', number=repetitions,
                                             setup="from __main__ import fi_normal_density, test_values_as_vectors")

    print(f"Fi 1/abs(1.4-x) result: {fi_one_over_x_result}")
    print(f"Fi 1/(qrt(pow((1.4 - x),2) result: {fi_one_over_sqrt_result}")
    print(f"Fi normal density result: {fi_normal_density_result}")
