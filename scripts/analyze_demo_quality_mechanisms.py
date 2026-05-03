from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from ilrl_lab.bc import load_bc_checkpoint, normalize_obs, predict_action
from ilrl_lab.envs import DetourWaypointVelocityAviary
from ilrl_lab.experts import detour_waypoint_velocity_expert
from ilrl_lab.ppo_training import (
    build_ppo_model,
    build_training_env,
    evaluate_model,
    initialize_actor_from_bc,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "figures" / "demo_quality_mechanisms"

CLEAN_DATASET = ROOT / "artifacts" / "datasets" / "detour_expert_20260502_172927.npz"
NOISY_DATASET = ROOT / "artifacts" / "datasets" / "demo_quality" / "noisy50" / "detour_noisy50_expert_20260504_013459.npz"
CLEAN_BC = ROOT / "artifacts" / "checkpoints" / "bc_aligned" / "clean50" / "bc_20260504_022538" / "checkpoint.pt"
NOISY_BC = ROOT / "artifacts" / "checkpoints" / "bc_aligned" / "noisy50" / "bc_20260504_022538" / "checkpoint.pt"


def sample_eval_states(episodes: int = 30, seed: int = 50000, max_states: int = 1500) -> np.ndarray:
    env = DetourWaypointVelocityAviary(gui=False)
    states: list[np.ndarray] = []
    for idx in range(episodes):
        obs, _ = env.reset(seed=seed + idx)
        while True:
            states.append(np.asarray(obs, dtype=np.float32))
            action = detour_waypoint_velocity_expert(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            if len(states) >= max_states:
                break
        if len(states) >= max_states:
            break
    env.close()
    return np.stack(states[:max_states])


def dataset_action_stats(dataset_path: Path) -> dict[str, object]:
    data = np.load(dataset_path)
    acts = data["acts"].astype(np.float32)
    episode_ids = data["episode_ids"].astype(np.int32)
    action_std = acts.std(axis=0)
    action_abs_mean = np.abs(acts).mean(axis=0)

    delta_norms = []
    for ep in np.unique(episode_ids):
        ep_acts = acts[episode_ids == ep]
        if len(ep_acts) >= 2:
            delta_norms.extend(np.linalg.norm(np.diff(ep_acts, axis=0), axis=1).tolist())

    return {
        "num_transitions": int(len(acts)),
        "action_std_per_dim": action_std.astype(float).tolist(),
        "action_abs_mean_per_dim": action_abs_mean.astype(float).tolist(),
        "mean_delta_action_norm": float(np.mean(delta_norms)) if delta_norms else 0.0,
        "p90_delta_action_norm": float(np.percentile(delta_norms, 90)) if delta_norms else 0.0,
    }


def bc_action_stats(checkpoint_path: Path, eval_states: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    device = torch.device("cpu")
    model, obs_mean, obs_std, _ = load_bc_checkpoint(checkpoint_path, device)
    actions = np.stack([predict_action(model, obs, obs_mean, obs_std, device) for obs in eval_states])
    return {
        "action_std_per_dim": actions.std(axis=0).astype(float).tolist(),
        "action_abs_mean_per_dim": np.abs(actions).mean(axis=0).astype(float).tolist(),
        "mean_pairwise_l2_to_mean": float(np.linalg.norm(actions - actions.mean(axis=0, keepdims=True), axis=1).mean()),
    }, actions


def compare_bc_actions(clean_actions: np.ndarray, noisy_actions: np.ndarray) -> dict[str, float]:
    delta = noisy_actions - clean_actions
    cosine = np.sum(clean_actions * noisy_actions, axis=1) / (
        np.linalg.norm(clean_actions, axis=1) * np.linalg.norm(noisy_actions, axis=1) + 1e-8
    )
    return {
        "mean_action_l2_diff": float(np.linalg.norm(delta, axis=1).mean()),
        "p90_action_l2_diff": float(np.percentile(np.linalg.norm(delta, axis=1), 90)),
        "mean_cosine_similarity": float(np.mean(cosine)),
    }


def ppo_mean_actions(model, obs_batch: np.ndarray) -> np.ndarray:
    obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        mean_actions = distribution.distribution.mean
    return mean_actions.detach().cpu().numpy()


class SaveCheckpointCallback(BaseCallback):
    def __init__(self, run_dir: Path, save_freq: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.run_dir = run_dir
        self.save_freq = int(save_freq)
        self.saved_paths: list[str] = []

    def _on_step(self) -> bool:
        if self.save_freq > 0 and self.num_timesteps % self.save_freq == 0:
            path = self.run_dir / f"checkpoint_{self.num_timesteps}.zip"
            self.model.save(path)
            self.saved_paths.append(str(path))
        return True


class EvalHistoryCallback(BaseCallback):
    def __init__(
        self,
        run_dir: Path,
        eval_freq: int,
        eval_episodes: int,
        eval_seed: int,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.run_dir = run_dir
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.eval_seed = int(eval_seed)
        self.obs_mean = np.asarray(obs_mean, dtype=np.float32)
        self.obs_std = np.asarray(obs_std, dtype=np.float32)
        self.history: list[dict[str, float | int]] = []

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            record, _ = evaluate_model(
                self.model,
                self.eval_episodes,
                self.eval_seed,
                task_variant="detour",
                obs_mean=self.obs_mean,
                obs_std=self.obs_std,
            )
            record.timesteps = int(self.num_timesteps)
            self.history.append(asdict(record))
        return True


class Args:
    def __init__(self, run_dir: Path):
        self.learning_rate = 3e-4
        self.n_steps = 1024
        self.batch_size = 256
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.ent_coef = 0.0
        self.vf_coef = 0.5
        self.log_std_init = -0.5
        self.run_dir = run_dir


def short_run_divergence(run_name: str, bc_checkpoint: Path, eval_states: np.ndarray, total_timesteps: int = 50000) -> dict[str, object]:
    run_dir = OUT_DIR / "ppo_divergence_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    bc_model, obs_mean, obs_std, _ = load_bc_checkpoint(bc_checkpoint, device)
    normalized_eval_states = normalize_obs(eval_states, obs_mean, obs_std)

    env = build_training_env(
        gui=False,
        seed=7,
        task_variant="detour",
        obs_mean=obs_mean,
        obs_std=obs_std,
    )
    args = Args(run_dir)
    model = build_ppo_model(env, seed=7, args=args)
    initialize_actor_from_bc(model, bc_checkpoint, torch.device("cpu"), bc_kl_coef=0.0003)

    checkpoint_cb = SaveCheckpointCallback(run_dir=run_dir, save_freq=10000)
    eval_cb = EvalHistoryCallback(
        run_dir=run_dir,
        eval_freq=10000,
        eval_episodes=20,
        eval_seed=60000,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )
    callback = CallbackList([checkpoint_cb, eval_cb])
    model.save(run_dir / "checkpoint_0.zip")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    model.save(run_dir / f"checkpoint_{total_timesteps}.zip")
    env.close()

    bc_actions = np.stack([predict_action(bc_model, obs, obs_mean, obs_std, device) for obs in eval_states])

    divergence = []
    for step in [0, 10000, 20000, 30000, 40000, 50000]:
        ckpt = run_dir / f"checkpoint_{step}.zip"
        ppo_model = type(model).load(ckpt)
        mean_actions = ppo_mean_actions(ppo_model, normalized_eval_states)
        l2 = np.linalg.norm(mean_actions - bc_actions, axis=1)
        cosine = np.sum(mean_actions * bc_actions, axis=1) / (
            np.linalg.norm(mean_actions, axis=1) * np.linalg.norm(bc_actions, axis=1) + 1e-8
        )
        divergence.append(
            {
                "timesteps": step,
                "mean_action_l2_from_bc": float(l2.mean()),
                "p90_action_l2_from_bc": float(np.percentile(l2, 90)),
                "mean_cosine_to_bc": float(np.mean(cosine)),
            }
        )

    out = {
        "run_name": run_name,
        "checkpoint_paths": [str(run_dir / f"checkpoint_{step}.zip") for step in [0, 10000, 20000, 30000, 40000, 50000]],
        "divergence": divergence,
        "eval_history": eval_cb.history,
    }
    (run_dir / "divergence_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def plot_summary(summary: dict[str, object]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))

    # 1. dataset action variance
    ax = axes[0, 0]
    clean_std = np.array(summary["dataset_action_stats"]["clean"]["action_std_per_dim"])
    noisy_std = np.array(summary["dataset_action_stats"]["noisy"]["action_std_per_dim"])
    x = np.arange(len(clean_std))
    width = 0.36
    ax.bar(x - width / 2, clean_std, width=width, label="Clean")
    ax.bar(x + width / 2, noisy_std, width=width, label="Noisy")
    ax.set_title("1. Demo action variance")
    ax.set_xticks(x)
    ax.set_xticklabels(["dx", "dy", "dz", "speed"])
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)

    # 2. BC action spread
    ax = axes[0, 1]
    clean_bc_std = np.array(summary["bc_action_stats"]["clean"]["action_std_per_dim"])
    noisy_bc_std = np.array(summary["bc_action_stats"]["noisy"]["action_std_per_dim"])
    ax.bar(x - width / 2, clean_bc_std, width=width, label="Clean BC")
    ax.bar(x + width / 2, noisy_bc_std, width=width, label="Noisy BC")
    ax.set_title("2. BC action spread on eval states")
    ax.set_xticks(x)
    ax.set_xticklabels(["dx", "dy", "dz", "speed"])
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)

    # 3. same-state action difference
    ax = axes[1, 0]
    metrics = summary["bc_action_comparison"]
    labels = ["Mean L2 diff", "P90 L2 diff", "Mean cosine"]
    vals = [metrics["mean_action_l2_diff"], metrics["p90_action_l2_diff"], metrics["mean_cosine_similarity"]]
    colors = ["#dc2626", "#ea580c", "#2563eb"]
    ax.bar(np.arange(len(labels)), vals, color=colors)
    ax.set_title("3. Clean vs noisy BC on same states")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=15)
    ax.grid(alpha=0.25, axis="y")

    # 4. PPO divergence
    ax = axes[1, 1]
    for name, color in [("clean", "#2563eb"), ("noisy", "#dc2626")]:
        divergence = summary["ppo_divergence"][name]["divergence"]
        steps = [d["timesteps"] for d in divergence]
        l2 = [d["mean_action_l2_from_bc"] for d in divergence]
        ax.plot(steps, l2, marker="o", label=f"{name} prior", color=color)
    ax.set_title("4. PPO moves away from BC prior")
    ax.set_xlabel("PPO timesteps")
    ax.set_ylabel("Mean action L2 from BC")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Why clean vs noisy demos behave differently", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mechanism_overview.png", dpi=180, bbox_inches="tight")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_stats = {
        "clean": dataset_action_stats(CLEAN_DATASET),
        "noisy": dataset_action_stats(NOISY_DATASET),
    }

    eval_states = sample_eval_states()
    clean_bc_stats, clean_bc_actions = bc_action_stats(CLEAN_BC, eval_states)
    noisy_bc_stats, noisy_bc_actions = bc_action_stats(NOISY_BC, eval_states)
    bc_comparison = compare_bc_actions(clean_bc_actions, noisy_bc_actions)

    clean_div = short_run_divergence("clean50_seed7_short", CLEAN_BC, eval_states)
    noisy_div = short_run_divergence("noisy50_seed7_short", NOISY_BC, eval_states)

    summary = {
        "dataset_action_stats": dataset_stats,
        "bc_action_stats": {
            "clean": clean_bc_stats,
            "noisy": noisy_bc_stats,
        },
        "bc_action_comparison": bc_comparison,
        "ppo_divergence": {
            "clean": clean_div,
            "noisy": noisy_div,
        },
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_summary(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
