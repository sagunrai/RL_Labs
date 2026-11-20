# Importing necessary libraries
import numpy as np
import torch
from dqn import ReplayBuffer, QNetwork, dqn_loss, train_dqn


def test_buffer_capacity_and_overwrite():
    b = ReplayBuffer(5, 4)
    for i in range(8):
        b.push(np.full(4, i), i % 2, float(i), np.full(4, i + 1), i == 7)
    assert len(b) == 5
    assert set(b.r.tolist()) == {3., 4., 5., 6., 7.}   # oldest (0,1,2) overwritten


def test_done_float_and_dtypes():
    b = ReplayBuffer(10, 4)
    b.push(np.zeros(4), 0, 1.0, np.ones(4), True)
    b.push(np.zeros(4), 1, 1.0, np.ones(4), False)
    s, a, r, s2, d = b.sample(64)
    assert d.dtype == torch.float32 and set(d.tolist()) <= {0.0, 1.0}
    assert a.dtype == torch.int64 and s.dtype == torch.float32


def test_uniform_sampling():
    b = ReplayBuffer(10, 4)
    for i in range(10):
        b.push(np.zeros(4), 0, float(i), np.zeros(4), False)
    r = torch.cat([b.sample(1000)[2] for _ in range(1000)])   # 10^6 draws
    counts = torch.bincount(r.long(), minlength=10).float() / r.numel()
    assert (counts - 0.1).abs().max() < 0.005, counts


def test_qnet_shape_no_softmax():
    q = QNetwork()
    out = q(torch.randn(7, 4))
    assert out.shape == (7, 2)
    assert not torch.allclose(out.sum(1), torch.ones(7))   # raw outputs, not probabilities


def test_loss_hand_computed_and_no_target_grad():
    online, target = QNetwork(hidden=8), QNetwork(hidden=8)
    s = torch.randn(3, 4); s2 = torch.randn(3, 4)
    a = torch.tensor([0, 1, 1]); r = torch.tensor([1., 1., 1.]); d = torch.tensor([0., 1., 0.])
    gamma = 0.9
    with torch.no_grad():
        y = r + gamma * (1 - d) * target(s2).max(1).values
        q = online(s)[torch.arange(3), a]
        expected = ((q - y) ** 2).mean()
    target.train()
    loss = dqn_loss(online, target, (s, a, r, s2, d), gamma)
    assert abs(loss.item() - expected.item()) < 1e-6
    assert not target.training
    loss.backward()
    assert all(p.grad is None for p in target.parameters())
    assert any(p.grad is not None for p in online.parameters())


def test_short_training_invariants():
    out = train_dqn(n_episodes=30, seed=0, warmup=200, target_sync=100)
    assert out["finite"] and out["sync_exact"] and len(out["sync_steps"]) > 0
    assert all(s % 100 == 0 for s in out["sync_steps"])


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
