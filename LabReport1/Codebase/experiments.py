"""
Task 5 experiments. Run: python experiments.py  -> figures in ./figures, results.json
"""

# Importing required libraries
import json, os
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qlearning import make_env, train, evaluate_greedy

N_EP = 20000
SEEDS = [0, 1, 2]
os.makedirs("figures", exist_ok=True)


def run(cfg):
    name, algo, seed, kw, slippery = cfg
    env = make_env(slippery=slippery)
    Q, R, L, snaps = train(env, algo, n_episodes=N_EP, seed=seed, **kw)
    return dict(name=name, algo=algo, seed=seed, Q=Q, R=R, L=L, snaps=snaps,
                greedy=evaluate_greedy(env, Q, 1000))


def smooth(x, w=500):
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def configs():
    cfgs = []
    for s in SEEDS:
        cfgs.append(("Q-learning", "q_learning", s, {}, True))
        cfgs.append(("SARSA", "sarsa", s, {}, True))
    for a in (0.02, 0.05, 0.3, 0.5):
        for s in SEEDS[:2]:
            cfgs.append((f"alpha={a}", "q_learning", s, dict(alpha=a), True))
    for e in (0.0, 0.1, 0.3):
        for s in SEEDS[:2]:
            cfgs.append((f"eps_end={e}", "q_learning", s, dict(eps_end=e), True))
    for g in (0.9, 0.95):
        for s in SEEDS[:2]:
            cfgs.append((f"gamma={g}", "q_learning", s, dict(gamma=g), True))
    for algo in ("q_learning", "sarsa"):
        for s in SEEDS[:2]:
            cfgs.append((f"deterministic {algo}", algo, s, {}, False))
    return cfgs


def group(res, name):
    return [r for r in res if r["name"] == name]


def curve_plot(ax, rs, label, key="R", w=500):
    ys = np.array([smooth(r[key], w) for r in rs])
    x = np.arange(ys.shape[1]) + w
    ax.plot(x, ys.mean(0), label=label)
    ax.fill_between(x, ys.min(0), ys.max(0), alpha=0.2)


def main():
    with Pool() as p:
        res = p.map(run, configs())

    # Fig 1: Q-learning vs SARSA
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for n in ("Q-learning", "SARSA"):
        curve_plot(axs[0], group(res, n), n)
        curve_plot(axs[1], group(res, n), n, key="L")
    axs[0].set(title="Success rate (500-ep moving avg)", xlabel="episode", ylabel="return")
    axs[1].set(title="Episode length", xlabel="episode", ylabel="steps")
    axs[0].legend(); fig.tight_layout(); fig.savefig("figures/fig1_qlearning_vs_sarsa.png", dpi=150)

    # Fig 2: hyperparameter sweeps
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
    for n in ("alpha=0.02", "alpha=0.05", "Q-learning", "alpha=0.3", "alpha=0.5"):
        curve_plot(axs[0], group(res, n), "alpha=0.1" if n == "Q-learning" else n)
    axs[0].set(title="Learning rate", xlabel="episode", ylabel="success rate")
    for n in ("eps_end=0.0", "Q-learning", "eps_end=0.1", "eps_end=0.3"):
        curve_plot(axs[1], group(res, n), "eps_end=0.01" if n == "Q-learning" else n)
    axs[1].set(title="Final exploration rate", xlabel="episode")
    for n in ("gamma=0.9", "gamma=0.95", "Q-learning"):
        curve_plot(axs[2], group(res, n), "gamma=0.99" if n == "Q-learning" else n)
    axs[2].set(title="Discount factor", xlabel="episode")
    for a in axs: a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/fig2_hyperparameters.png", dpi=150)

    # Fig 3: slippery vs deterministic
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    curve_plot(ax, group(res, "Q-learning"), "slippery Q-learning")
    curve_plot(ax, group(res, "deterministic q_learning"), "deterministic Q-learning")
    curve_plot(ax, group(res, "deterministic sarsa"), "deterministic SARSA")
    ax.set(title="Effect of stochastic dynamics", xlabel="episode", ylabel="success rate")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("figures/fig3_slippery_vs_deterministic.png", dpi=150)

    # Fig 4: value function and policy (seed 0 Q-learning vs SARSA) + snapshots
    arrows = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # LEFT DOWN RIGHT UP (gym order)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, n in zip(axs, ("Q-learning", "SARSA")):
        Q = [r for r in group(res, n) if r["seed"] == 0][0]["Q"]
        V = Q.max(1).reshape(8, 8)
        im = ax.imshow(V, cmap="viridis")
        for s in range(64):
            i, j = divmod(s, 8)
            if Q[s].max() > 0:
                dx, dy = arrows[int(np.argmax(Q[s]))]
                ax.arrow(j, i, 0.3 * dx, 0.3 * dy, head_width=0.12, color="w")
        ax.set_title(f"{n}: max_a Q(s,a) and greedy policy"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig("figures/fig4_value_policy.png", dpi=150)

    snaps = [r for r in group(res, "Q-learning") if r["seed"] == 0][0]["snaps"]
    pick = [snaps[i] for i in (0, 2, 7, len(snaps) - 1)]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (ep, Q) in zip(axs, pick):
        ax.imshow(Q.max(1).reshape(8, 8), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"episode {ep}"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig("figures/fig5_q_snapshots.png", dpi=150)

    # Summary table
    names = sorted({r["name"] for r in res})
    summary = {}
    for n in names:
        rs = group(res, n)
        tr = [r["R"][-1000:].mean() for r in rs]
        gr = [r["greedy"] for r in rs]
        summary[n] = dict(train_last1000=float(np.mean(tr)), greedy=float(np.mean(gr)),
                          n_runs=len(rs))
        print(f"{n:28s} train(last1000)={np.mean(tr):.3f}  greedy={np.mean(gr):.3f}")
    json.dump(summary, open("results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
