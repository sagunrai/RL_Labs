"""
Tabular Q-learning and SARSA on FrozenLake (numpy + gymnasium only).
"""

# Importing necessary libraries
import numpy as np
import gymnasium as gym


def make_env(slippery=True, map_name="8x8", max_steps=200):
    return gym.make("FrozenLake-v1", map_name=map_name, is_slippery=slippery,
                    max_episode_steps=max_steps)


# Task 1
def epsilon_greedy(Q, s, eps, rng):
    """Random action w.p. eps, else greedy; argmax ties broken uniformly."""
    n_actions = Q.shape[1]
    if rng.random() < eps:
        return int(rng.integers(n_actions))
    q = Q[s]
    best = np.flatnonzero(q == q.max())
    return int(rng.choice(best))


# Task 2
def q_learning_update(Q, s, a, r, s_next, terminal, alpha, gamma):
    """Off-policy update. terminal=True -> zero bootstrap. Returns new Q(s,a)."""
    target = r if terminal else r + gamma * np.max(Q[s_next])
    Q[s, a] += alpha * (target - Q[s, a])
    return Q[s, a]


# Task 3
def sarsa_update(Q, s, a, r, s_next, a_next, terminal, alpha, gamma):
    """On-policy update using the action actually taken next."""
    target = r if terminal else r + gamma * Q[s_next, a_next]
    Q[s, a] += alpha * (target - Q[s, a])
    return Q[s, a]


# Task 4
def train(env, algo="q_learning", n_episodes=30000, alpha=0.1, gamma=0.99,
          eps_start=1.0, eps_end=0.01, decay_episodes=None, seed=0,
          snapshot_every=1000):
    """Run training.
    Epsilon decays linearly from eps_start to eps_end over `decay_episodes`
    episodes (default: 80% of n_episodes), then stays at eps_end.
    Returns: Q, returns (n_episodes,), lengths (n_episodes,),
             snapshots (list of (episode, Q copy)).
    """
    assert algo in ("q_learning", "sarsa")
    rng = np.random.default_rng(seed)
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    if decay_episodes is None:
        decay_episodes = int(0.8 * n_episodes)

    returns = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=int)
    snapshots = []

    for ep in range(n_episodes):
        frac = min(ep / max(decay_episodes, 1), 1.0)
        eps = eps_start + frac * (eps_end - eps_start)

        s, _ = env.reset(seed=seed if ep == 0 else None)
        a = epsilon_greedy(Q, s, eps, rng)
        G, t, done = 0.0, 0, False
        while not done:
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            a2 = epsilon_greedy(Q, s2, eps, rng)
            # Only true termination zeroes the bootstrap; a time-limit
            # truncation is not a terminal state.
            if algo == "q_learning":
                q_learning_update(Q, s, a, r, s2, terminated, alpha, gamma)
            else:
                sarsa_update(Q, s, a, r, s2, a2, terminated, alpha, gamma)
            s, a = s2, a2
            G += r
            t += 1
        returns[ep], lengths[ep] = G, t
        if (ep + 1) % snapshot_every == 0:
            snapshots.append((ep + 1, Q.copy()))

    return Q, returns, lengths, snapshots


def success_rate(returns, last=1000):
    return float(np.mean(returns[-last:]))


def evaluate_greedy(env, Q, n_episodes=2000, seed=123):
    """Success rate of the pure greedy policy (eps=0)."""
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n_episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = epsilon_greedy(Q, s, 0.0, rng)
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc
        wins += r
    return wins / n_episodes
