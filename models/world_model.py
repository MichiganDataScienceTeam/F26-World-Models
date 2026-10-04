from pathlib import Path

import numpy as np
import torch
from torch import nn

from envs.common import broadcast_inputs

PRETRAINED = Path(__file__).parent / "pretrained"


class WorldModel(nn.Module):
    """forward predicts normalized changes; predict returns next observations.

    The drone target passes through unchanged and is excluded from the inputs.
    """

    def __init__(self, environment, width=64):
        super().__init__()
        if environment not in ("drone", "pendulum"):
            raise ValueError("environment must be drone or pendulum")

        self.environment = environment
        self.width = width
        self.state_size, input_size = (6, 9) if environment == "drone" else (3, 4)
        self.net = nn.Sequential(
            nn.Linear(input_size, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, self.state_size))

        self.register_buffer("xm", torch.zeros(input_size))
        self.register_buffer("xs", torch.ones(input_size))
        self.register_buffer("ym", torch.zeros(self.state_size))
        self.register_buffer("ys", torch.ones(self.state_size))
        self.register_buffer("residual_rms", torch.zeros(self.state_size))

    def input_features(self, observations, actions):
        sizes, action_size = ((6, 8), 2) if self.environment == "drone" else ((3,), 1)
        obs, action = broadcast_inputs(observations, actions, sizes, action_size)

        if self.environment == "drone":
            action = np.clip(action, 0, 1)
            return np.concatenate((obs[..., :2], np.sin(obs[..., 2:3]), np.cos(obs[..., 2:3]),
                                   obs[..., 3:6], action), axis=-1)
        return np.concatenate((obs, np.clip(action, -2, 2)), axis=-1)

    def forward(self, raw_features):
        return self.net((raw_features - self.xm) / self.xs)

    def _next_observation(self, observations, delta):
        obs = np.asarray(observations)
        obs = np.broadcast_to(obs, delta.shape[:-1] + (obs.shape[-1],))
        return np.concatenate((obs[..., :self.state_size] + delta, obs[..., self.state_size:]), axis=-1)

    def _hidden_features(self, observations, actions):
        x = torch.as_tensor(self.input_features(observations, actions),
                            dtype=self.xm.dtype, device=self.xm.device)
        return self.net[:-1]((x - self.xm) / self.xs)

    @torch.inference_mode()
    def features(self, observations, actions):
        """Last hidden layer with a constant feature for the intercept."""

        hidden = self._hidden_features(observations, actions)
        return torch.cat((hidden, torch.ones_like(hidden[..., :1])), dim=-1).cpu().numpy()

    @torch.inference_mode()
    def predict_with_features(self, observations, actions):
        """Return next observations and features in one network pass."""

        hidden = self._hidden_features(observations, actions)
        delta = (self.net[-1](hidden)*self.ys + self.ym).cpu().numpy()
        phi = torch.cat((hidden, torch.ones_like(hidden[..., :1])), dim=-1).cpu().numpy()
        return self._next_observation(observations, delta), phi

    @torch.inference_mode()
    def predict(self, observations, actions, return_std=False):
        hidden = self._hidden_features(observations, actions)
        delta = (self.net[-1](hidden)*self.ys + self.ym).cpu().numpy()
        mean = self._next_observation(observations, delta)
        if not return_std:
            return mean

        # Held-out error
        physical_std = np.broadcast_to(self.residual_rms.cpu().numpy(), delta.shape)
        std = np.concatenate((physical_std, np.zeros_like(mean[..., self.state_size:])), axis=-1)
        return mean, std

    def set_normalization(self, x, y):
        with torch.no_grad():
            self.xm.copy_(x.mean(0))
            self.xs.copy_(x.std(0).clamp_min(1e-6))
            self.ym.copy_(y.mean(0))
            self.ys.copy_(y.std(0).clamp_min(1e-6))

    def save(self, path):
        torch.save({"environment": self.environment, "width": self.width,
                    "state_dict": self.state_dict()}, path)


def load_pretrained(environment):
    if environment not in ("drone", "pendulum"):
        raise ValueError("environment must be drone or pendulum")
    path = PRETRAINED / f"{environment}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"No {environment} checkpoint. Run python -m pretraining.pendulum")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("environment", environment) != environment:
        raise ValueError("Checkpoint is for a different environment")

    model = WorldModel(environment, checkpoint["width"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().requires_grad_(False)
