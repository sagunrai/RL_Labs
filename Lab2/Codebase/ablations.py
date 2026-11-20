"""
Task 5 ablations
"""
import json, os
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_EP = 500
os.makedirs("figures", exist_ok=True)

CONFIGS = {  # name -> (kwargs, seeds)
    "baseline": ({}, [0, 1, 2]),
    "no target net": (dict(use_target=False), [0, 1, 2]),
    "no replay": (dict(use_replay=False), [0, 1, 2]),
    "sync=50": (dict(target_sync=50), [0, 1]),
    "sync=1000": (dict(target_sync=1000), [0, 1]),
    "buffer=1000": (dict(buffer_capacity=1000), [0, 1]),
    "gamma=0.9": (dict(gamma=0.9), [0, 1]),
}


def run(job):
    import torch
    torch.set_num_threads(1)
    from dqn import train_dqn
    name, kw, seed = job
    o = train_dqn(n_episodes=N_EP, seed=seed, **kw)
    return dict(name=name, seed=seed, returns=o["returns"], ma100=o["ma100"],
                best=o["best_ma100"], losses=o["losses"], mean_q=o["mean_q"],
                finite=o["finite"], steps=o["total_steps"])


def grp(res, n):
    return [r for r in res if r["name"] == n]


def plot_ma(ax, rs, label):
    ys = np.array([r["ma100"] for r in rs])
    x = np.arange(1, ys.shape[1] + 1)
    ax.plot(x, ys.mean(0), label=label)
    ax.fill_between(x, ys.min(0), ys.max(0), alpha=0.2)


def main():
    jobs = [(n, kw, s) for n, (kw, seeds) in CONFIGS.items() for s in seeds]
    with Pool(2) as p:
        res = p.map(run, jobs, chunksize=1)

    def finish(ax, title):
        ax.axhline(475, ls="--", c="gray", lw=0.8)
        ax.set(title=title, xlabel="episode", ylabel="100-episode moving avg return")
        ax.legend(fontsize=8)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for r in grp(res, "baseline"):
        ax.plot(r["returns"], alpha=0.25, lw=0.7)
    plot_ma(ax, grp(res, "baseline"), "baseline MA100 (3 seeds)")
    finish(ax, "DQN baseline on CartPole-v1")
    fig.tight_layout(); fig.savefig("figures/fig1_baseline.png", dpi=150)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for n in ("baseline", "no target net"):
        plot_ma(axs[0], grp(res, n), n)
    finish(axs[0], "Ablation: target network")
    for n in ("baseline", "no replay"):
        plot_ma(axs[1], grp(res, n), n)
    finish(axs[1], "Ablation: experience replay")
    fig.tight_layout(); fig.savefig("figures/fig2_target_replay.png", dpi=150)

    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    for n in ("sync=50", "baseline", "sync=1000"):
        plot_ma(axs[0], grp(res, n), "sync=200" if n == "baseline" else n)
    finish(axs[0], "Target sync period (steps)")
    for n in ("buffer=1000", "baseline"):
        plot_ma(axs[1], grp(res, n), "buffer=50000" if n == "baseline" else n)
    finish(axs[1], "Replay buffer capacity")
    for n in ("gamma=0.9", "baseline"):
        plot_ma(axs[2], grp(res, n), "gamma=0.99" if n == "baseline" else n)
    finish(axs[2], "Discount factor")
    fig.tight_layout(); fig.savefig("figures/fig3_sweeps.png", dpi=150)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for n in ("baseline", "no target net", "no replay"):
        r = grp(res, n)[0]
        k = 200
        sm = np.convolve(r["losses"], np.ones(k) / k, mode="valid")
        axs[0].plot(sm, label=n)
        axs[1].plot(np.arange(len(r["mean_q"])) * 100, r["mean_q"], label=n)
    axs[0].set(title="TD loss (seed 0, smoothed)", xlabel="update", yscale="log")
    axs[1].set(title="Mean Q on sampled states (seed 0)", xlabel="env step")
    axs[0].legend(fontsize=8); axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/fig4_loss_q.png", dpi=150)

    summary = {}
    for n in CONFIGS:
        rs = grp(res, n)
        summary[n] = dict(best_ma100_mean=float(np.mean([r["best"] for r in rs])),
                          best_ma100_per_seed=[float(r["best"]) for r in rs],
                          final_ma100_mean=float(np.mean([r["ma100"][-1] for r in rs])),
                          solved_475=bool(any(r["best"] >= 475 for r in rs)),
                          all_finite=bool(all(r["finite"] for r in rs)),
                          total_steps_mean=float(np.mean([r["steps"] for r in rs])))
        print(f"{n:14s} best={summary[n]['best_ma100_mean']:.1f} "
              f"final={summary[n]['final_ma100_mean']:.1f} finite={summary[n]['all_finite']}")
    json.dump(summary, open("ablation_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
