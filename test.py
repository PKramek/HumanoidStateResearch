import pytest

from util import calculate_shaped_reward


@pytest.mark.parametrize(
    "reward, fi_value, last_fi_value, gamma, expected",
    [
        (0, 0, 0, 1.0, 0),
        (0.0, 500.0, 500.0, 0.99, -5.0),
        (0.0, -500.0, -500.0, 0.99, 5.0)
    ])
def test_eval(reward, fi_value, last_fi_value, gamma, expected):
    output = calculate_shaped_reward(
        reward=reward, fi_value=fi_value, last_fi_value=last_fi_value,
        gamma=gamma)
    assert output == expected
