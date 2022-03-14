import argparse
from typing import List

import gym
import matplotlib.pyplot as plt


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


parser = argparse.ArgumentParser(description='Display information about state vector of gym environment')
parser.add_argument('--env', type=str, help='Name of gym environment', required=True)
parser.add_argument('--steps', type=int, help='Number of steps to perform in the environment', default=1000)
parser.add_argument('--plot-index', type=int, help='Index of value in observation vector to create plot of', default=1)
parser.add_argument('--plot-path', type=str, help='Path to directory, where created plots will be created')

# According to this thread: https://github.com/openai/gym/issues/585
# first 3 elements of the state vector are (X,Y,Z) coordinates

if __name__ == '__main__':
    args = parser.parse_args()

    try:
        env = gym.make(args.env)
    except gym.error.Error as e:
        raise ValueError(f"Unknown environment '{args.env}'") from e

    observation = env.reset()
    print(f"Length of observation: {len(observation)}")
    print(f"First observation: {observation}")

    values = [observation[args.plot_index]]

    for t in range(args.steps):
        print(f"Observation[plot_index]={observation[args.plot_index]}")

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        values.append(observation[args.plot_index])
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
    env.close()

    if args.plot_path is not None:
        plot_and_save_values(args.env, args.plot_index, args.plot_path, values)
