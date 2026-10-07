"""The networks of runs 9 to 12 and their training (PyTorch; the agents run the same maths in numpy).

Two kinds:
- ``plain``: one MLP applied to each row on its own (each route, or each release group): features -> share of capacity.
  ``layers`` hidden layers of ``hidden`` units (runs 1 to 8: 2 x 64).
- ``shared`` (run 11, flows only): the routes of one week decide together, in two steps.
  1. encoder: each route's features -> a summary of ``summary`` numbers;
  2. sharing: for each route, the mean summary of the routes with the same source, the same destination and the same
     good (``SlotFeatures.share_matrices``: three (slots x slots) averaging matrices);
  3. decoder: [its summary, the three means] -> share of capacity.
  So each route knows what the routes next to it are about to do before it decides.

Both train with the asymmetric Huber loss (answering less than the teacher counts ``under_weight`` times more), keep
the epoch with the best loss on the held-out episodes, and export numpy arrays named as the agent reads them.
"""

import numpy as np
import torch


def mlp(n_in: int, hidden: int, layers: int, n_out: int) -> torch.nn.Sequential:
    """``layers`` hidden layers of ``hidden`` units with ReLU, then a linear output of ``n_out``."""
    parts, width = [], n_in
    for _ in range(layers):
        parts += [torch.nn.Linear(width, hidden), torch.nn.ReLU()]
        width = hidden
    parts.append(torch.nn.Linear(width, n_out))
    return torch.nn.Sequential(*parts)


class Shared(torch.nn.Module):
    """Encoder -> mean summaries over same source, destination and good -> decoder (module docstring)."""

    def __init__(self, n_in, hidden, layers, summary, share):
        super().__init__()
        self.encoder = mlp(n_in, hidden, layers, summary)
        self.decoder = mlp(4 * summary, hidden, layers, 1)
        self.register_buffer("share", torch.tensor(share, dtype=torch.float32))  # (3, slots, slots)

    def forward(self, x):  # x: (weeks, slots, features) -> (weeks, slots)
        h = torch.relu(self.encoder(x))
        means = [torch.einsum("ij,wjk->wik", m, h) for m in self.share]
        return self.decoder(torch.cat([h, *means], dim=-1))[..., 0]


def _loss(under_weight):
    huber = torch.nn.HuberLoss(delta=0.5, reduction="none")

    def loss(pred, target):
        return (huber(pred, target) * torch.where(pred < target, under_weight, 1.0)).mean()

    return loss


def _scaler(X, is_val):
    flat = X[~is_val].reshape(-1, X.shape[-1])
    mean, std = flat.mean(0), flat.std(0)
    return mean.astype(np.float32), np.where(std < 1e-3, 1.0, std).astype(np.float32)


def _evaluate(net, Xv, Yv, loss_fn, chunk=262_144):
    """Loss and mean absolute error on the held-out rows, in chunks (a large set may not fit a GPU at once)."""
    loss_sum = abs_sum = 0.0
    with torch.no_grad():
        for i in range(0, len(Xv), chunk):
            pred, y = net(Xv[i : i + chunk]), Yv[i : i + chunk]
            loss_sum += loss_fn(pred, y).item() * len(y)
            abs_sum += (pred.clamp(min=0) - y).abs().sum().item()
    n = max(len(Xv), 1)
    return loss_sum / n, abs_sum / (n * (Yv[0].numel() if len(Yv) else 1))


def _fit(net, Xt, Yt, Xv, Yv, epochs, lr, batch, under_weight, label, patience=None):
    """Adam on the training rows; keeps the epoch with the best held-out loss; stops after ``patience`` epochs
    without improvement (None: never)."""
    if len(Xt) == 0 or len(Xv) == 0:
        raise ValueError("training needs at least 2 episodes: some to learn from, some held out")
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = _loss(under_weight)
    best, best_state, history, since_best = float("inf"), None, [], 0
    device = next(net.parameters()).device
    for epoch in range(1, epochs + 1):
        net.train()
        order = torch.randperm(len(Xt), device=device)
        total = 0.0
        for i in range(0, len(Xt), batch):
            idx = order[i : i + batch]
            loss = loss_fn(net(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        net.eval()
        val, val_mae = _evaluate(net, Xv, Yv, loss_fn)
        history.append({"epoch": epoch, "train_loss": total / len(Xt), "val_loss": val, "val_mae": val_mae})
        marker = ""
        since_best += 1
        if val < best:
            best, marker, since_best = val, "  <- best so far", 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        print(
            f"  {label} epoch {epoch:3d}  train {total / len(Xt):.4f}  validation {val:.4f}  "
            f"(mean error {val_mae:.3f} x capacity){marker}",
            flush=True,
        )
        if patience is not None and since_best >= patience:
            print(f"  {label}: no better held-out loss for {patience} epochs, stopping", flush=True)
            break
    net.load_state_dict(best_state)
    return history


def train_plain(
    X,
    Y,
    episode,
    val_episodes,
    *,
    module,
    hidden,
    layers,
    epochs,
    lr,
    batch,
    seed,
    under_weight,
    label,
    device="cpu",
    patience=None,
):
    """A plain MLP on rows (features -> share); its numpy arrays and history. ``device``: "cpu" or "cuda"."""
    torch.manual_seed(seed)
    is_val = np.isin(episode, val_episodes)  # whole episodes held out, so validation is honest
    mean, std = _scaler(X, is_val)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    Xt, Yt = t(module.normalise(X[~is_val], mean, std)), t(Y[~is_val])
    Xv, Yv = t(module.normalise(X[is_val], mean, std)), t(Y[is_val])
    net = mlp(X.shape[1], hidden, layers, 1).to(device)
    wrapped = torch.nn.Sequential(net, torch.nn.Flatten(0))  # (rows, 1) -> (rows,)
    history = _fit(wrapped, Xt, Yt, Xv, Yv, epochs, lr, batch, under_weight, label, patience)
    return {"mean": mean, "std": std, **mlp_arrays(net)}, history


def train_shared(
    X, Y, episode, val_episodes, share, *, module, hidden, layers, summary, epochs, lr, batch, seed, under_weight, label
):
    """The shared network on weeks: X (rows, F) and Y (rows,) hold the slots of each week in order."""
    torch.manual_seed(seed)
    S = share.shape[1]
    Xw, Yw, ew = X.reshape(-1, S, X.shape[1]), Y.reshape(-1, S), episode.reshape(-1, S)[:, 0]
    is_val = np.isin(ew, val_episodes)
    mean, std = _scaler(Xw, is_val)
    t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    Xt, Yt = t(module.normalise(Xw[~is_val], mean, std)), t(Yw[~is_val])
    Xv, Yv = t(module.normalise(Xw[is_val], mean, std)), t(Yw[is_val])
    net = Shared(X.shape[1], hidden, layers, summary, share)
    history = _fit(net, Xt, Yt, Xv, Yv, epochs, lr, max(1, batch // S), under_weight, label)
    arrays = {"mean": mean, "std": std}
    arrays |= {f"enc_{k}": v for k, v in mlp_arrays(net.encoder).items()}
    arrays |= {f"dec_{k}": v for k, v in mlp_arrays(net.decoder).items()}
    return arrays, history


def mlp_arrays(net: torch.nn.Sequential) -> dict:
    """W1, b1, ..., Wn, bn of an MLP, as numpy (x @ W + b)."""
    layers = [m for m in net if isinstance(m, torch.nn.Linear)]
    out = {}
    for i, layer in enumerate(layers, start=1):
        out[f"W{i}"] = layer.weight.detach().cpu().numpy().T.astype(np.float32)
        out[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    return out
