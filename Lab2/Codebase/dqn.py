"""
DQN on CartPole-v1 (torch + numpy + gymnasium).
"""

# Importing necessary libraries
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


# Task 1
class ReplayBuffer:
    """Fixed-capacity circular buffer of (s, a, r, s', done) transitions."""

    def __init__(self, capacity, state_dim=4, seed=0):
        self.capacity = int(capacity)
        self.s = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(self.capacity, dtype=np.int64)
        self.r = np.zeros(self.capacity, dtype=np.float32)
        self.s2 = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.d = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def push(self, s, a, r, s2, done):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i] = s, a, r, s2
        self.d[i] = 1.0 if done else 0.0          # bool -> 0.0 / 1.0
        self.ptr = (self.ptr + 1) % self.capacity  # overwrite oldest when full
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)  # uniform, with replacement
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx]),
                torch.from_numpy(self.d[idx]))

    def __len__(self):
        return self.size

# Task 2
class QNetwork(nn.Module):
    """MLP obs -> hidden -> hidden -> Q-values (ReLU, raw linear output, He init)."""

    def __init__(self, obs_dim=4, n_actions=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# Task 3
def dqn_loss(online, target, batch, gamma):
    """MSE between y = r + gamma (1-done) max_a' Q_target(s',a') and Q_online(s,a)."""
    s, a, r, s2, d = batch
    target.eval()
    with torch.no_grad():
        y = r + gamma * (1.0 - d) * target(s2).max(dim=1).values
    q = online(s).gather(1, a.long().unsqueeze(1)).squeeze(1)
    return nn.functional.mse_loss(q, y)


# Task 4
def train_dqn(n_episodes=500, seed=0, gamma=0.99, lr=5e-4, batch_size=64,
              buffer_capacity=50_000, eps_start=1.0, eps_end=0.05,
              eps_decay_steps=3_000, target_sync=200, warmup=500,
              use_target=True, use_replay=True, grad_clip=10.0,
              env_id="CartPole-v1", verbose=False):
    """
    Train DQN. Returns dict with returns, lengths, losses, qnet, diagnostics.
    use_target=False : bootstrap from the online net (ablation).
    use_replay=False : update only on the latest transition (ablation).
    Target net is hard-synced (exact copy) whenever total env steps % target_sync == 0.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env = gym.make(env_id)
    env.action_space.seed(seed)
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n

    online, target = QNetwork(obs_dim, n_actions), QNetwork(obs_dim, n_actions)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    buf = ReplayBuffer(buffer_capacity, obs_dim, seed)

    returns, lengths, losses, mean_q = [], [], [], []
    sync_steps, sync_ok = [], []
    step = 0
    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed if ep == 0 else None)
        G, t, done = 0.0, 0, False
        while not done:
            eps = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay_steps)
            if rng.random() < eps:
                a = int(rng.integers(n_actions))
            else:
                with torch.no_grad():
                    a = int(online(torch.as_tensor(s, dtype=torch.float32)).argmax())
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            # time-limit truncation is not a true terminal -> keep bootstrapping
            buf.push(s, a, r, s2, terminated)
            s = s2
            G += r; t += 1; step += 1

            if len(buf) >= max(warmup, batch_size):
                if use_replay:
                    batch = buf.sample(batch_size)
                else:
                    i = (buf.ptr - 1) % buf.capacity
                    batch = tuple(torch.from_numpy(x[i:i + 1]) for x in
                                  (buf.s, buf.a, buf.r, buf.s2, buf.d))
                tgt = target if use_target else online
                if not use_target:
                    # bootstrap from the online net itself (no frozen copy)
                    with torch.no_grad():
                        ref = QNetwork(obs_dim, n_actions)
                        ref.load_state_dict(online.state_dict())
                    tgt = ref
                loss = dqn_loss(online, tgt, batch, gamma)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), grad_clip)
                opt.step()
                losses.append(loss.item())
                if step % 100 == 0:
                    with torch.no_grad():
                        mean_q.append(online(batch[0]).mean().item())

            if step % target_sync == 0:
                target.load_state_dict(online.state_dict())
                sync_steps.append(step)
                sync_ok.append(all(torch.equal(p, q) for p, q in
                                   zip(online.parameters(), target.parameters())))
        returns.append(G); lengths.append(t)
        if verbose and (ep + 1) % 25 == 0:
            print(f"ep {ep+1} MA100={np.mean(returns[-100:]):.1f} eps={eps:.2f}", flush=True)

    env.close()
    returns = np.array(returns)
    ma = np.array([returns[max(0, i - 99): i + 1].mean() for i in range(len(returns))])
    finite = all(torch.isfinite(p).all().item() for p in online.parameters()) and \
        bool(np.all(np.isfinite(losses)))
    return dict(returns=returns, lengths=np.array(lengths), losses=np.array(losses),
                mean_q=np.array(mean_q), ma100=ma, best_ma100=float(ma[min(99, len(ma) - 1):].max()),
                qnet=online, target=target, sync_steps=sync_steps,
                sync_exact=all(sync_ok), finite=finite, total_steps=step)
