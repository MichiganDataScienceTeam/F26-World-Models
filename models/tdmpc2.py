"""A pretrained TD-MPC2 agent (Hansen, Su & Wang 2024, https://github.com/nicklashansen/tdmpc2): the world model
and its MPPI planner, for state-based DMControl tasks, loading the official 5M-parameter checkpoints. Adapted from
TD-MPC2's code (license at the end of this file).

The planner can plan through a command map: the motors are assumed to deliver a = clip(M c + b) for command c,
so the planner chooses commands that produce the actions it wants. M = I, b = 0 is the original planner.
"""

import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

CHECKPOINTS = "https://huggingface.co/nicklashansen/tdmpc2/resolve/main/dmcontrol/{}-1.pt"
CACHE = Path.home() / ".cache" / "f26-world-models"

# TD-MPC2's configuration for the 5M models (tdmpc2/config.yaml)
LATENT, HIDDEN, ENCODER_HIDDEN, SIMNORM_GROUP = 512, 512, 256, 8
NUM_Q, NUM_BINS, VMIN, VMAX = 5, 101, -10., 10.
LOG_STD_MIN, LOG_STD_MAX = -10., 2.
ITERATIONS, SAMPLES, ELITES, POLICY_SAMPLES = 6, 512, 64, 24
MIN_STD, MAX_STD, TEMPERATURE = .05, 2., .5


class NormedLinear(nn.Linear):
    """Linear, then LayerNorm, then Mish (or SimNorm)."""

    def __init__(self, n_in, n_out, act="mish"):
        super().__init__(n_in, n_out)
        self.ln, self.act = nn.LayerNorm(n_out), act

    def forward(self, x):
        x = self.ln(super().forward(x))
        if self.act == "simnorm":  # softmax within groups of 8
            return F.softmax(x.unflatten(-1, (-1, SIMNORM_GROUP)), -1).flatten(-2)
        return F.mish(x)


def mlp(n_in, n_out, last="linear", n_hidden=HIDDEN):
    last_layer = nn.Linear(n_hidden, n_out) if last == "linear" else NormedLinear(n_hidden, n_out, last)
    return nn.Sequential(NormedLinear(n_in, n_hidden), NormedLinear(n_hidden, n_hidden), last_layer)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class TDMPC2(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.action_size = action_size
        self._encoder = nn.ModuleDict({"state": nn.Sequential(
            NormedLinear(obs_size, ENCODER_HIDDEN), NormedLinear(ENCODER_HIDDEN, LATENT, "simnorm"))})
        self._dynamics = mlp(LATENT + action_size, LATENT, "simnorm")
        self._reward = mlp(LATENT + action_size, NUM_BINS)
        self._pi = mlp(LATENT, 2 * action_size)
        # The Q ensemble: each layer's parameters stacked over the 5 heads
        q_shapes = [(HIDDEN, LATENT + action_size), (HIDDEN,), (HIDDEN,), (HIDDEN,),
                    (HIDDEN, HIDDEN), (HIDDEN,), (HIDDEN,), (HIDDEN,), (NUM_BINS, HIDDEN), (NUM_BINS,)]
        self.q = nn.ParameterList([nn.Parameter(torch.zeros(NUM_Q, *s)) for s in q_shapes])
        self.register_buffer("bins", torch.linspace(VMIN, VMAX, NUM_BINS))

    def encode(self, obs):
        return self._encoder["state"](obs)

    def next(self, z, a):
        return self._dynamics(torch.cat([z, a], -1))

    def to_scalar(self, logits):
        """Rewards and values are predicted as distributions over bins (in symlog space)."""
        return symexp((F.softmax(logits, -1) * self.bins).sum(-1, keepdim=True))

    def reward(self, z, a):
        return self.to_scalar(self._reward(torch.cat([z, a], -1)))

    def pi(self, z):
        """An action sampled from the policy prior."""
        mean, log_std = self._pi(z).chunk(2, -1)
        log_std = LOG_STD_MIN + .5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(log_std) + 1)
        return torch.tanh(mean + torch.randn_like(mean) * log_std.exp())

    def Q(self, z, a):
        """Average of two randomly chosen heads of the Q ensemble (as TD-MPC2's planner uses it)."""
        heads = torch.randperm(NUM_Q, device=z.device)[:2]
        p = [w[heads] for w in self.q]
        x = torch.cat([z, a], -1).expand(2, *z.shape[:-1], -1)
        for i in (0, 4):
            x = torch.baddbmm(p[i + 1][:, None], x.flatten(1, -2), p[i].transpose(1, 2)).view(*x.shape[:-1], -1)
            x = F.mish(F.layer_norm(x, (HIDDEN,)) * p[i + 2][:, None] + p[i + 3][:, None])
        logits = torch.baddbmm(p[9][:, None], x.flatten(1, -2), p[8].transpose(1, 2)).view(*x.shape[:-1], -1)
        return self.to_scalar(logits).mean(0)


def load(task="walker-walk", obs_size=24, action_size=6, device="cpu"):
    """The official checkpoint, downloaded once (30 MB) to ~/.cache/f26-world-models."""
    path = CACHE / f"{task}-1.pt"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"downloading the TD-MPC2 {task} checkpoint to {path}")
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(CHECKPOINTS.format(task), partial)
        partial.rename(path)                                          # only complete downloads are kept
    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    state = {k.replace("_Qs.params.", "q."): v for k, v in state.items() if not k.startswith("_target_Qs")}
    state["bins"] = torch.linspace(VMIN, VMAX, NUM_BINS)
    model = TDMPC2(obs_size, action_size)
    model.load_state_dict(state)
    return model.eval().to(device)


class Planner:
    """TD-MPC2's MPPI for B agents in one batched pass. Agent k plans through its own command map
    a = clip(M[k] c + b[k]); set M and b before calling act."""

    def __init__(self, model, B=1, horizon=3, discount=.99, seed=0):
        self.model, self.B, self.horizon, self.discount = model, B, horizon, discount
        A, dev = model.action_size, model.bins.device
        self.M = torch.eye(A, device=dev).repeat(B, 1, 1)
        self.b = torch.zeros(B, A, device=dev)
        self.prev_mean = torch.zeros(B, horizon, A, device=dev)
        torch.manual_seed(seed)

    def actions(self, c):
        """What the motors deliver for commands c (B, n, A)."""
        return (c @ self.M.transpose(1, 2) + self.b[:, None]).clamp(-1, 1)

    def policy_commands(self, z):
        """Policy-prior actions turned into the commands that produce them."""
        B, n = z.shape[:2]
        a = self.model.pi(z.flatten(0, 1)).view(B, n, -1)
        return ((a - self.b[:, None]) @ self.M_inv.transpose(1, 2)).clamp(-1, 1)

    def value(self, z, c):
        """Discounted model rewards over the horizon, then the value of the final latent; z (B, N, L)."""
        B, N = z.shape[:2]
        G, discount, z = 0, 1, z.flatten(0, 1)
        for t in range(self.horizon):
            a = self.actions(c[t]).flatten(0, 1)
            G = G + discount * self.model.reward(z, a).view(B, N, 1)
            z = self.model.next(z, a)
            discount *= self.discount
        return G + discount * self.model.Q(z, self.model.pi(z)).view(B, N, 1)

    @torch.no_grad()
    def act(self, obs, first):
        """Commands (B, A) for observations (B, obs_size); first[k] marks agent k's first step of an episode."""
        m, B, H, A = self.model, self.B, self.horizon, self.model.action_size
        dev = m.bins.device
        self.M_inv = torch.linalg.pinv(self.M.cpu(), rtol=1e-2).to(dev)
        z = m.encode(torch.as_tensor(obs, dtype=torch.float32, device=dev))
        c = torch.empty(H, B, SAMPLES, A, device=dev)
        zp = z[:, None].expand(B, POLICY_SAMPLES, -1)
        for t in range(H):  # trajectories from the policy prior
            c[t, :, :POLICY_SAMPLES] = self.policy_commands(zp)
            if t < H - 1:
                B_, n = zp.shape[:2]
                zp = m.next(zp.flatten(0, 1), self.actions(c[t, :, :POLICY_SAMPLES]).flatten(0, 1)).view(B_, n, -1)
        z = z[:, None].expand(B, SAMPLES, -1)
        mean = torch.zeros(B, H, A, device=dev)
        for k in range(B):
            if not first[k]:
                mean[k, :-1] = self.prev_mean[k, 1:]
        std = torch.full((B, H, A), MAX_STD, device=dev)
        for _ in range(ITERATIONS):
            noise = torch.randn(H, B, SAMPLES - POLICY_SAMPLES, A, device=dev)
            c[:, :, POLICY_SAMPLES:] = (mean.transpose(0, 1)[:, :, None]
                                        + std.transpose(0, 1)[:, :, None] * noise).clamp(-1, 1)
            value = self.value(z, c).nan_to_num(0)
            idx = torch.topk(value[..., 0], ELITES, dim=1).indices
            elite_value = torch.gather(value, 1, idx[..., None])
            elite = torch.gather(c, 2, idx[None, :, :, None].expand(H, B, ELITES, A))
            w = torch.exp(TEMPERATURE * (elite_value - elite_value.max(1, keepdim=True).values))
            w = (w / w.sum(1, keepdim=True))[None]
            mean = ((w * elite).sum(2) / (w.sum(2) + 1e-9)).transpose(0, 1)
            std = ((w * (elite - mean.transpose(0, 1)[:, :, None])**2).sum(2) / (w.sum(2) + 1e-9)).sqrt()
            std = std.transpose(0, 1).clamp(MIN_STD, MAX_STD)
        gumbel = -torch.empty_like(w[0, ..., 0]).exponential_().log()  # sample one elite per agent
        j = (w[0, ..., 0].log() + gumbel).argmax(-1)
        self.prev_mean.copy_(mean)
        return elite[0, torch.arange(B, device=dev), j].clamp(-1, 1).cpu().numpy()


# TD-MPC2 license:
# MIT License
#
# Copyright (c) Nicklas Hansen (2023).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
