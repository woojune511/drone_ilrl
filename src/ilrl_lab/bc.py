from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class BehaviorCloningPolicy(nn.Module):
    """Small MLP policy for waypoint imitation learning."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


def normalize_obs(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (obs - mean) / std


def predict_action(
    model: BehaviorCloningPolicy,
    obs: np.ndarray,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    normalized = normalize_obs(obs.astype(np.float32), obs_mean, obs_std)
    obs_tensor = torch.from_numpy(normalized).to(device=device, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = model(obs_tensor).squeeze(0).cpu().numpy()
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def save_bc_checkpoint(
    path: Path,
    model: BehaviorCloningPolicy,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "obs_mean": obs_mean.astype(np.float32),
        "obs_std": obs_std.astype(np.float32),
        "metadata": metadata,
    }
    torch.save(payload, path)


def load_bc_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[BehaviorCloningPolicy, np.ndarray, np.ndarray, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    metadata = payload["metadata"]
    model = BehaviorCloningPolicy(
        obs_dim=int(metadata["obs_dim"]),
        action_dim=int(metadata["action_dim"]),
        hidden_sizes=tuple(int(x) for x in metadata["hidden_sizes"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    obs_mean = payload["obs_mean"].astype(np.float32)
    obs_std = payload["obs_std"].astype(np.float32)
    return model, obs_mean, obs_std, metadata
