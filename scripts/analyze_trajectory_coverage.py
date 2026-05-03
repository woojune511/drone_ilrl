from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ilrl_lab.bc import load_bc_checkpoint, predict_action
from ilrl_lab.envs import DetourWaypointVelocityAviary
from ilrl_lab.experts import detour_waypoint_velocity_expert


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "figures" / "trajectory_coverage_analysis"

CONFIG = {
    10: {
        "dataset_npz": ROOT / "artifacts" / "datasets" / "demo_qty" / "q10" / "detour_expert_20260503_234921.npz",
        "bc_checkpoint": ROOT / "artifacts" / "checkpoints" / "bc_demo_qty" / "q10" / "bc_20260503_235025" / "checkpoint.pt",
    },
    50: {
        "dataset_npz": ROOT / "artifacts" / "datasets" / "detour_expert_20260502_172927.npz",
        "bc_checkpoint": ROOT / "artifacts" / "checkpoints" / "bc" / "bc_20260502_172938" / "checkpoint.pt",
    },
    200: {
        "dataset_npz": ROOT / "artifacts" / "datasets" / "demo_qty" / "q200" / "detour_expert_20260503_235013.npz",
        "bc_checkpoint": ROOT / "artifacts" / "checkpoints" / "bc_demo_qty" / "q200" / "bc_20260503_235025" / "checkpoint.pt",
    },
}

LAYOUT = {
    "wall_x": 0.0,
    "gap_center_y": 0.50,
    "gap_half_width": 0.18,
    "wall_half_thickness": 0.05,
    "entry_x": -0.18,
    "exit_x": 0.22,
    "corridor_y_tolerance": 0.10,
    "exit_margin_x": 0.03,
}


def detour_stage(pos: np.ndarray, goal: np.ndarray) -> str:
    if goal[0] <= LAYOUT["wall_x"]:
        return "goal"
    corridor_aligned = abs(float(pos[1]) - LAYOUT["gap_center_y"]) <= LAYOUT["corridor_y_tolerance"]
    if pos[0] < LAYOUT["wall_x"] - LAYOUT["wall_half_thickness"]:
        if not corridor_aligned:
            return "entry"
        if pos[0] < LAYOUT["exit_x"] - LAYOUT["exit_margin_x"]:
            return "exit"
    elif pos[0] < LAYOUT["exit_x"] - LAYOUT["exit_margin_x"]:
        return "exit"
    return "goal"


def load_expert_trajectories(dataset_path: Path):
    data = np.load(dataset_path)
    obs = data["obs"].astype(np.float32)
    episode_ids = data["episode_ids"].astype(np.int32)
    episode_goals = data["episode_goals"].astype(np.float32)
    episode_lengths = data["episode_lengths"].astype(np.int32)

    trajectories: list[dict[str, np.ndarray]] = []
    for episode_idx, _ in enumerate(episode_lengths):
        mask = episode_ids == episode_idx
        positions = obs[mask, 0:3].astype(np.float32)
        goal = episode_goals[episode_idx].astype(np.float32)
        trajectories.append({"positions": positions, "goal": goal})
    return trajectories


def draw_wall(ax):
    y_min = -0.75
    y_max = 0.75
    gap_low = LAYOUT["gap_center_y"] - 0.18
    gap_high = LAYOUT["gap_center_y"] + 0.18
    ax.plot([0.0, 0.0], [y_min, gap_low], color="#b91c1c", linewidth=5, alpha=0.8)
    ax.plot([0.0, 0.0], [gap_high, y_max], color="#b91c1c", linewidth=5, alpha=0.8)
    ax.axhline(LAYOUT["gap_center_y"], color="#94a3b8", linestyle=":", linewidth=1)


def plot_expert_xy_overlay(all_trajs: dict[int, list[dict[str, np.ndarray]]]):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    for ax, quantity in zip(axes, [10, 50, 200]):
        for traj in all_trajs[quantity]:
            xy = traj["positions"][:, :2]
            ax.plot(xy[:, 0], xy[:, 1], color="#2563eb", alpha=0.15, linewidth=1)
            ax.scatter([xy[0, 0]], [xy[0, 1]], color="#1d4ed8", s=8, alpha=0.3)
            ax.scatter([traj["goal"][0]], [traj["goal"][1]], color="#059669", s=8, alpha=0.25, marker="x")
        draw_wall(ax)
        ax.set_title(f"{quantity} demos")
        ax.set_xlabel("x")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("y")
    axes[0].set_xlim(-0.72, 0.72)
    axes[0].set_ylim(-0.12, 0.72)
    fig.suptitle("Expert Trajectory Coverage (XY overlay)", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "expert_xy_overlay.png", dpi=180, bbox_inches="tight")


def plot_stage_heatmaps(all_trajs: dict[int, list[dict[str, np.ndarray]]]):
    stages = ["entry", "exit", "goal"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    bins = 45
    xedges = np.linspace(-0.72, 0.72, bins)
    yedges = np.linspace(-0.12, 0.72, bins)
    for row, quantity in enumerate([10, 50, 200]):
        stage_points = {stage: [] for stage in stages}
        for traj in all_trajs[quantity]:
            goal = traj["goal"]
            for pos in traj["positions"]:
                stage_points[detour_stage(pos, goal)].append(pos[:2])
        for col, stage in enumerate(stages):
            ax = axes[row, col]
            pts = np.asarray(stage_points[stage], dtype=np.float32) if stage_points[stage] else np.zeros((0, 2), dtype=np.float32)
            if len(pts) > 0:
                hist, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[xedges, yedges])
                ax.imshow(
                    hist.T,
                    origin="lower",
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    cmap="viridis",
                    aspect="auto",
                )
            draw_wall(ax)
            if row == 0:
                ax.set_title(stage)
            if col == 0:
                ax.set_ylabel(f"{quantity} demos\ny")
            if row == 2:
                ax.set_xlabel("x")
    fig.suptitle("Stage-wise Expert Visitation Heatmaps", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stage_heatmaps.png", dpi=180, bbox_inches="tight")


def path_length(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def resample_xy_path(positions: np.ndarray, num_points: int = 100) -> np.ndarray:
    xy = positions[:, :2]
    if len(xy) == 1:
        return np.repeat(xy, num_points, axis=0)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = cumulative[-1]
    if total < 1e-8:
        return np.repeat(xy[:1], num_points, axis=0)
    targets = np.linspace(0.0, total, num_points)
    out = []
    for t in targets:
        idx = np.searchsorted(cumulative, t, side="right") - 1
        idx = min(idx, len(xy) - 2)
        denom = cumulative[idx + 1] - cumulative[idx]
        alpha = 0.0 if denom < 1e-8 else (t - cumulative[idx]) / denom
        out.append((1.0 - alpha) * xy[idx] + alpha * xy[idx + 1])
    return np.stack(out)


def circular_std(angles: np.ndarray) -> float:
    if len(angles) == 0:
        return 0.0
    c = np.mean(np.cos(angles))
    s = np.mean(np.sin(angles))
    r = np.sqrt(c * c + s * s)
    if r <= 1e-8:
        return float(np.pi / np.sqrt(3))
    return float(np.sqrt(-2.0 * np.log(r)))


def compute_diversity_metrics(all_trajs: dict[int, list[dict[str, np.ndarray]]]):
    summary: dict[int, dict[str, float]] = {}
    for quantity, trajs in all_trajs.items():
        lengths = [path_length(t["positions"]) for t in trajs]
        corridor_y_stds = []
        approach_heading_stds = []
        resampled = []
        for traj in trajs:
            pos = traj["positions"]
            goal = traj["goal"]
            exit_mask = np.array([detour_stage(p, goal) == "exit" for p in pos], dtype=bool)
            goal_mask = np.array([detour_stage(p, goal) == "goal" and p[0] > LAYOUT["exit_x"] - 0.05 for p in pos], dtype=bool)
            corridor_y_stds.append(float(np.std(pos[exit_mask, 1])) if exit_mask.any() else 0.0)
            if goal_mask.sum() >= 3:
                deltas = np.diff(pos[goal_mask][:, :2], axis=0)
                headings = np.arctan2(deltas[:, 1], deltas[:, 0])
                approach_heading_stds.append(circular_std(headings))
            else:
                approach_heading_stds.append(0.0)
            resampled.append(resample_xy_path(pos))

        pairwise = []
        for a, b in combinations(resampled, 2):
            pairwise.append(float(np.linalg.norm(a - b, axis=1).mean()))
        summary[quantity] = {
            "mean_path_length": float(np.mean(lengths)),
            "std_path_length": float(np.std(lengths)),
            "mean_corridor_y_std": float(np.mean(corridor_y_stds)),
            "mean_final_approach_heading_std": float(np.mean(approach_heading_stds)),
            "mean_pairwise_path_distance": float(np.mean(pairwise)) if pairwise else 0.0,
        }
    return summary


def plot_diversity_metrics(metric_summary: dict[int, dict[str, float]]):
    quantities = [10, 50, 200]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(len(quantities))

    axes[0].bar(x, [metric_summary[q]["mean_path_length"] for q in quantities], color="#2563eb")
    axes[0].set_title("Mean Path Length")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(q) for q in quantities])
    axes[0].set_xlabel("Demo episodes")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(x, [metric_summary[q]["mean_pairwise_path_distance"] for q in quantities], color="#7c3aed")
    axes[1].set_title("Mean Pairwise Path Distance")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(q) for q in quantities])
    axes[1].set_xlabel("Demo episodes")
    axes[1].grid(alpha=0.25, axis="y")

    axes[2].bar(x, [metric_summary[q]["mean_final_approach_heading_std"] for q in quantities], color="#059669")
    axes[2].set_title("Final Approach Heading Std")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([str(q) for q in quantities])
    axes[2].set_xlabel("Demo episodes")
    axes[2].grid(alpha=0.25, axis="y")

    fig.suptitle("Expert Trajectory Diversity Metrics", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "path_diversity_metrics.png", dpi=180, bbox_inches="tight")


def rollout_policy(checkpoint_path: Path, seed: int, episodes: int = 5):
    device = torch.device("cpu")
    model, obs_mean, obs_std, _ = load_bc_checkpoint(checkpoint_path, device)
    env = DetourWaypointVelocityAviary(gui=False)
    trajectories = []
    for idx in range(episodes):
        obs, info = env.reset(seed=seed + idx)
        positions = [np.asarray(info["position"], dtype=np.float32)]
        goal = np.asarray(info["goal"], dtype=np.float32)
        success = False
        while True:
            action = predict_action(model, obs, obs_mean, obs_std, device)
            obs, _, terminated, truncated, info = env.step(action)
            positions.append(np.asarray(info["position"], dtype=np.float32))
            if terminated or truncated:
                success = bool(info["success"])
                break
        trajectories.append({"positions": np.stack(positions), "goal": goal, "success": success})
    env.close()
    return trajectories


def rollout_expert(seed: int, episodes: int = 5):
    env = DetourWaypointVelocityAviary(gui=False)
    trajectories = []
    for idx in range(episodes):
        obs, info = env.reset(seed=seed + idx)
        positions = [np.asarray(info["position"], dtype=np.float32)]
        goal = np.asarray(info["goal"], dtype=np.float32)
        success = False
        while True:
            action = detour_waypoint_velocity_expert(obs)
            obs, _, terminated, truncated, info = env.step(action)
            positions.append(np.asarray(info["position"], dtype=np.float32))
            if terminated or truncated:
                success = bool(info["success"])
                break
        trajectories.append({"positions": np.stack(positions), "goal": goal, "success": success})
    env.close()
    return trajectories


def plot_bc_vs_expert_overlays():
    eval_seed = 300
    expert_trajs = rollout_expert(eval_seed, episodes=5)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    bc_rollout_summary: dict[int, dict[str, float]] = {}

    for ax, quantity in zip(axes, [10, 50, 200]):
        bc_trajs = rollout_policy(CONFIG[quantity]["bc_checkpoint"], eval_seed, episodes=5)
        expert_success = float(np.mean([t["success"] for t in expert_trajs]))
        bc_success = float(np.mean([t["success"] for t in bc_trajs]))
        bc_rollout_summary[quantity] = {
            "expert_success_5eval": expert_success,
            "bc_success_5eval": bc_success,
        }

        for traj in expert_trajs:
            xy = traj["positions"][:, :2]
            ax.plot(xy[:, 0], xy[:, 1], color="#059669", alpha=0.55, linewidth=2)
        for traj in bc_trajs:
            xy = traj["positions"][:, :2]
            ax.plot(xy[:, 0], xy[:, 1], color="#dc2626", alpha=0.7, linewidth=1.6, linestyle="--")
        draw_wall(ax)
        ax.set_title(f"{quantity} demos")
        ax.set_xlabel("x")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("y")
    axes[0].set_xlim(-0.72, 0.72)
    axes[0].set_ylim(-0.12, 0.72)
    fig.suptitle("BC-only Rollouts vs Expert Rollouts (same eval seeds)", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bc_vs_expert_overlay.png", dpi=180, bbox_inches="tight")
    return bc_rollout_summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expert_trajs = {quantity: load_expert_trajectories(cfg["dataset_npz"]) for quantity, cfg in CONFIG.items()}

    plot_expert_xy_overlay(expert_trajs)
    plot_stage_heatmaps(expert_trajs)
    metric_summary = compute_diversity_metrics(expert_trajs)
    plot_diversity_metrics(metric_summary)
    bc_vs_expert_summary = plot_bc_vs_expert_overlays()

    summary = {
        "trajectory_diversity": metric_summary,
        "bc_vs_expert_rollout_summary": bc_vs_expert_summary,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
