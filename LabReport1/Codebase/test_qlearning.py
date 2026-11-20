# Importing necessary libraries
import numpy as np
from qlearning import (epsilon_greedy, q_learning_update, sarsa_update, make_env, train, success_rate, evaluate_greedy)


def test_tie_breaking():
    Q = np.zeros((1, 4))
    rng = np.random.default_rng(0)
    counts = np.bincount([epsilon_greedy(Q, 0, 0.0, rng) for _ in range(8000)], minlength=4)
    assert counts.min() > 1700, counts
    Q[0] = [0, 1, 1, 0]
    acts = {epsilon_greedy(Q, 0, 0.0, rng) for _ in range(500)}
    assert acts == {1, 2}


def test_eps_one_is_uniform():
    Q = np.zeros((1, 4)); Q[0, 2] = 5
    rng = np.random.default_rng(1)
    counts = np.bincount([epsilon_greedy(Q, 0, 1.0, rng) for _ in range(8000)], minlength=4)
    assert counts.min() > 1700


def test_q_update_hand_computed():
    # alpha=0.5, gamma=0.9
    Q = np.zeros((3, 2))
    Q[1] = [0.2, 0.6]
    q_learning_update(Q, 0, 1, 0.0, 1, False, 0.5, 0.9)   # target 0.54 -> 0.27
    assert abs(Q[0, 1] - 0.27) < 1e-8
    q_learning_update(Q, 1, 1, 1.0, 2, True, 0.5, 0.9)    # terminal: target 1 -> 0.8
    assert abs(Q[1, 1] - 0.8) < 1e-8
    Q[2] = [100, 100]                                      # must be ignored at terminal
    q_learning_update(Q, 1, 0, 0.0, 2, True, 0.5, 0.9)    # target 0 -> 0.1
    assert abs(Q[1, 0] - 0.1) < 1e-8


def test_sarsa_update_hand_computed():
    Q = np.zeros((3, 2))
    Q[1] = [0.2, 0.6]
    sarsa_update(Q, 0, 1, 0.0, 1, 0, False, 0.5, 0.9)     # uses Q[1,0]=0.2 -> 0.09
    assert abs(Q[0, 1] - 0.09) < 1e-8
    sarsa_update(Q, 0, 0, 1.0, 1, 1, True, 0.5, 0.9)      # terminal -> 0.5
    assert abs(Q[0, 0] - 0.5) < 1e-8


def test_success_rate_seed0():
    env = make_env()
    Q, R, L, snaps = train(env, "q_learning", seed=0)
    sr = success_rate(R, 1000)
    print("train success (last 1000):", sr, "greedy:", evaluate_greedy(env, Q))
    assert sr >= 0.7
    assert len(snaps) > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
