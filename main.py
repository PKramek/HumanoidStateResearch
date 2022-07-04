import argparse
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

# Description of Humanoid-v1 state vector, it supposedly was not changed
# https://github.com/openai/gym/wiki/Humanoid-V1
from util import normal_dist_density, calculate_shaped_reward


# According to this thread: https://github.com/openai/gym/issues/585
# first 3 elements of the state vector are (X,Y,Z) coordinates


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


def for_seminary(x: np.ndarray):
    index = 0
    return 0.0 if 1.3 < x[index] < 1.5 else -np.power((1.4 - x[index]) * 100, 2)


def for_seminary_correct_results(x: np.ndarray):
    index = 0
    return (10 * normal_dist_density(x[index], 1.4, 0.05)) + 500


def new_normal(x: np.ndarray):
    index = 0
    return (160 * normal_dist_density(x[index], 1.4, 1))


def new_normal_narrow(x: np.ndarray):
    index = 0
    return (3200 * normal_dist_density(x[index], 1.4, 0.05))


def new_normal_narrow_small_diff(x: np.ndarray):
    index = 0
    return (1600 * normal_dist_density(x[index], 1.4, 0.05))


def new_normal_narrow_big_diff(x: np.ndarray):
    index = 0
    return (6400 * normal_dist_density(x[index], 1.4, 0.05))


def new_normal_super_narrow(x: np.ndarray):
    index = 0
    return (6400 * normal_dist_density(x[index], 1.4, 0.02))


def new_normal_super_narrow_big_diff(x: np.ndarray):
    index = 0
    return (12800 * normal_dist_density(x[index], 1.4, 0.02))


def new_normal_super_narrow_small_diff(x: np.ndarray):
    index = 0
    return (3200 * normal_dist_density(x[index], 1.4, 0.02))


def func_normal_super_narrow(x, middle_of_normal_dist: float = 1.4):
    index = 0
    return 6400 * normal_dist_density(x[index], middle_of_normal_dist, 0.02)


def psi_normal(x, middle_of_normal_dist: float = 1.4):
    index = 0
    return 32 * normal_dist_density(x[index], middle_of_normal_dist, 0.05)


def psi_slightly_narrow(x, middle_of_normal_dist: float = 1.4):
    index = 0
    return 110 * normal_dist_density(x[index], middle_of_normal_dist, 0.015)


def psi_very_narrow(x, middle_of_normal_dist: float = 1.4):
    index = 0
    return 155 * normal_dist_density(x[index], middle_of_normal_dist, 0.01)


def psi_slightly_narrow_penalty(x, middle_of_normal_dist: float = 1.4):
    index = 0
    return 110 * normal_dist_density(x[index], middle_of_normal_dist, 0.015) - 5.18


def psi_slightly_narrow_penalty_five(x, middle_of_normal_dist: float = 1.4):
    return psi_slightly_narrow_penalty(x, middle_of_normal_dist) * 5


def psi_slightly_narrow_penalty_fifty(x, middle_of_normal_dist: float = 1.4):
    return psi_slightly_narrow_penalty(x, middle_of_normal_dist) * 50


def psi_less_narrow(x, middle_of_normal_dist: float = 1.4):
    index = 0
    return 55 * normal_dist_density(x[index], middle_of_normal_dist, 0.03)


def psi_less_narrow_penalty(x, middle_of_normal_dist: float = 1.4):
    index = 0
    return 55 * normal_dist_density(x[index], middle_of_normal_dist, 0.03) - 5.18


def alive_reward(x: np.ndarray):
    return 500.0


def alive_penalty(x: np.ndarray):
    return -500.0


def abs_penalty(x: np.ndarray, optimal_point: float = 1.4):
    index = 0
    return (- np.abs((x[index] - optimal_point))) * 25


def best_psi(state: np.ndarray, middle_of_normal_dist: float = 1.4):
    index = 0
    return 110 * normal_dist_density(state[index], middle_of_normal_dist, 0.015) - 5.18


def best_psi_shifted_down(state: np.ndarray, middle_of_normal_dist: float = 1.4):
    return best_psi(state, middle_of_normal_dist) - 5


def flipped_best_psi(state: np.ndarray, middle_of_normal_dist: float = 1.4):
    return - best_psi(state, middle_of_normal_dist)


def best_psi_ten(state: np.ndarray, middle_of_normal_dist: float = 1.4):
    return best_psi(state, middle_of_normal_dist) * 10


def squared_penalty(state: np.ndarray, optimal_point: float = 1.4):
    index = 0
    return - np.square((state[index] - optimal_point)) * 120


def abs_third_power(state: np.ndarray, optimal_point: float = 1.4):
    index = 0
    return - np.abs(np.power((state[index] - optimal_point), 3)) * 600


def abs_fourth_power(state: np.ndarray, optimal_point: float = 1.4):
    index = 0
    return - np.abs(np.power((state[index] - optimal_point), 4)) * 600


def second_power(state: np.ndarray, optimal_point: float = 1.4):
    index = 0
    return - np.power(abs((state[index] - optimal_point)), 2) * 100


def half_power(state: np.ndarray, optimal_point: float = 1.4):
    index = 0
    return - np.power(abs((state[index] - optimal_point)), 0.5) * 10


def linear_penalty(state: np.ndarray, optimal_point: float = 1.4):
    index = 0
    return - (abs(state[index] - optimal_point)) * 100


def t_student_penalty_df_0_2_scale_0_0_5(state: np.ndarray, optimal_point: float = 1.4):
    index = 0
    df = 0.2
    scale = 0.05
    loc = 1.4

    return t.pdf(state[index], df=df, scale=scale, loc=loc)


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
    plt.title(f"{fi.__name__}\nRewards prim from start of the episode till the fall")
    plt.ylabel("Rewards prim when passing from consequent states")
    plt.xlabel("State[0] (Mass center height)")
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
    plt.title(f"{fi.__name__}\nRewards prim from fall state to beginning point")
    plt.ylabel("Rewards prim when passing from consequent states")
    plt.xlabel("State[0] (Mass center height)")
    plt.grid()
    plt.show()

    return from_fall_to_beginning, rewards_prim


parser = argparse.ArgumentParser(description='Display information about state vector of gym environment')

parser.add_argument('--env', type=str, help='Name of gym environment', required=True)
parser.add_argument('--steps', type=int, help='Number of steps to perform in the environment', default=1000)
parser.add_argument('--plot-index', type=int, help='Index of value in observation vector to create plot of', default=1)
parser.add_argument('--plot-path', type=str, help='Path to directory, where created plots will be created')

if __name__ == '__main__':
    # TODO - Where does this weird peak comes from (using flipped_best_psi) ??

    used_fi = t_student_penalty_df_0_2_scale_0_0_5
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

    # plot_good_to_bad_states_reward_shaping_transition(fi=used_fi, step_size=step_size, base_reward=base_reward,
    #                                                   gamma=0.95)
    #
    # plot_bad_to_good_states_reward_shaping_transition(fi=used_fi, step_size=step_size, base_reward=base_reward,
    #                                                   gamma=0.95)

    functions_and_titles = [
        (used_fi, f"{used_fi.__name__}"),
        # (squared_penalty, f"{squared_penalty.__name__}"),
        # (abs_third_power, f"{abs_third_power.__name__}"),
        # (abs_fourth_power, f"{abs_fourth_power.__name__}")
        (second_power, f"{second_power.__name__}"),
        (half_power, f"{half_power.__name__}")

        # (squared_penalty_narrow, f"{squared_penalty_narrow.__name__}")
        # (psi_slightly_narrow_penalty, "Psi slightly narrow penalty"),
        # (psi_slightly_narrow_penalty_five, "Psi slightly narrow penalty Five"),
        # (psi_slightly_narrow_penalty_fifty, "Psi slightly narrow penalty Fifty"),
        # (psi_less_narrow_penalty, "Psi less narrow penalty"),
        # (abs_penalty, "ABS")
    ]

    # for function, plot_title in functions_and_titles:
    #     test_fi_x = np.array(list(map(function, test_values_as_vectors)))
    #     plot_fi_x(test_values, test_fi_x, plot_title)
