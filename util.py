from typing import List, Callable

import numpy as np
from matplotlib import pyplot as plt

from main import used_fi


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


def get_left_to_right_reward_values_in_reward_shaping(
        fi: Callable, input_x: np.ndarray,
        gamma: float, base_reward: float = 10.0):
    assert isinstance(input_x, np.ndarray)
    output_x = np.zeros_like(input_x)
    last_fi_value = fi([input_x[0]])

    output_x[0] = calculate_shaped_reward(base_reward, last_fi_value, last_fi_value, gamma)

    history = []

    for i in range(1, len(input_x)):
        fi_value = fi([input_x[i]])
        output_x[i] = calculate_shaped_reward(
            reward=base_reward, last_fi_value=last_fi_value,
            fi_value=fi_value, gamma=gamma
        )

        history.append(
            f"x_t={input_x[i]:.3f}, x_t_1={input_x[i - 1]:.3f} r'={output_x[i]:.3f}, r={base_reward:.3f}, last_fi_value={last_fi_value:.3f},gamma={gamma:.3f}, fi_value={fi_value:.3f}"
            f"\t{output_x[i]:.3f}={base_reward:.3f} - {last_fi_value:.3f} + {gamma:.3f}*{fi_value:.3f}")

        last_fi_value = fi_value

        # reward = reward - self._last_fi_value + self._gamma * fi_value

    return output_x, history


def get_right_to_left_reward_values_in_reward_shaping(
        fi: Callable, input_x: np.ndarray,
        gamma: float, base_reward: float = 10.0):
    assert isinstance(input_x, np.ndarray)
    reversed_input_x = np.flip(input_x)
    reversed_output, history = get_left_to_right_reward_values_in_reward_shaping(fi, reversed_input_x, gamma,
                                                                                 base_reward)

    return np.flip(reversed_output), history[::-1]


def plot_good_to_bad_states_reward_shaping_transition(fi: callable, step_size: float = 0.01, base_reward: float = 0.0,
                                                      gamma: float = 0.99):
    # TODO check this function
    left_to_right_input = np.arange(1.4, 1.6, step_size)
    right_to_left_input = np.arange(1.2, 1.4, step_size)

    left_to_right_output, left_to_right_history = get_left_to_right_reward_values_in_reward_shaping(
        fi=fi,
        input_x=left_to_right_input,
        gamma=gamma,
        base_reward=base_reward
    )

    right_to_left_output, right_to_left_history = get_right_to_left_reward_values_in_reward_shaping(
        fi=fi,
        input_x=right_to_left_input,
        gamma=gamma,
        base_reward=base_reward
    )

    whole_input = np.concatenate((right_to_left_input, left_to_right_input))
    whole_output = np.concatenate((right_to_left_output, left_to_right_output))

    plt.plot(whole_input, whole_output)
    plt.title(f"{used_fi.__name__}: From good to bad states")
    plt.show()


def plot_rewards_prim_from_beginning_to_fall(fi: callable, step_size: float = 0.01,
                                             base_reward: float = 0.0, gamma: float = 0.99):
    from_beginning_to_fall_range = np.arange(1.4, 1.2, -step_size)
    rewards_prim = np.zeros_like(from_beginning_to_fall_range)

    last_fi_value = fi([from_beginning_to_fall_range[0]])

    rewards_prim[0] = calculate_shaped_reward(reward=base_reward, last_fi_value=last_fi_value,
                                              fi_value=last_fi_value, gamma=gamma)

    for i in range(1, len(from_beginning_to_fall_range)):
        fi_value = fi([from_beginning_to_fall_range[i]])

        rewards_prim[i] = calculate_shaped_reward(reward=base_reward, last_fi_value=last_fi_value,
                                                  fi_value=fi_value, gamma=gamma)

    plt.plot(from_beginning_to_fall_range, rewards_prim)
    # plt.title(f"{fi.__name__}\nRewards prim from start of the episode till the fall")
    plt.title(f"Ukształtowane nagrody, przy tranzycji ze stanów lepszych do gorszych\n{fi.__name__}")
    plt.ylabel("Ukształtowane nagrody")
    plt.xlabel("Stan[0] (Wysokość środka cieżkości)")
    plt.grid()
    plt.show()

    return from_beginning_to_fall_range, rewards_prim


def plot_rewards_prim_from_fall_to_beginning(fi: callable, step_size: float = 0.01,
                                             base_reward: float = 0.0, gamma: float = 0.99):
    from_fall_to_beginning = np.arange(1.2, 1.4, step_size)
    rewards_prim = np.zeros_like(from_fall_to_beginning)

    last_fi_value = fi([from_fall_to_beginning[0]])

    rewards_prim[0] = calculate_shaped_reward(reward=base_reward, last_fi_value=last_fi_value,
                                              fi_value=last_fi_value, gamma=gamma)

    for i in range(1, len(from_fall_to_beginning)):
        fi_value = fi([from_fall_to_beginning[i]])

        rewards_prim[i] = calculate_shaped_reward(reward=base_reward, last_fi_value=last_fi_value,
                                                  fi_value=fi_value, gamma=gamma)

    plt.plot(from_fall_to_beginning, rewards_prim)
    # plt.title(f"{fi.__name__}\nUkształtowane nagrody, przy tranzycji ze stanów gorszych do lepszych")
    plt.title(f"Ukształtowane nagrody, przy tranzycji ze stanów gorszych do lepszych\n{fi.__name__}")
    plt.ylabel("Ukształtowane nagrody")
    plt.xlabel("Stan[0] (Wysokość środka cieżkości)")
    plt.grid()
    plt.show()

    return from_fall_to_beginning, rewards_prim
