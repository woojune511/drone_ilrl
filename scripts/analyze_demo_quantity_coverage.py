from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ilrl_lab.bc import load_bc_checkpoint, predict_action
from ilrl_lab.envs import DetourWaypointVelocityAviary


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "figures" / "demo_quantity_coverage"

CONFIG = {
    10: {
        "dataset_npz": ROOT / "artifacts" / "datasets" / "demo_qty" / "q10" / "detour_expert_20260503_234921.npz",
        "dataset_summary": ROOT / "artifacts" / "datasets" / "demo_qty" / "q10" / "detour_expert_20260503_234921_summary.json",
        "bc_checkpoint": ROOT / "artifacts" / "checkpoints" / "bc_demo_qty" / "q10" / "bc_20260503_235025" / "checkpoint.pt",
        "bc_eval": ROOT / "artifacts" / "evals" / "demo_qty" / "q10" / "bc_eval_20260503_235126.json",
    },
    50: {
        "dataset_npz": ROOT / "artifacts" / "datasets" / "detour_expert_20260502_172927.npz",
        "dataset_summary": ROOT / "artifacts" / "datasets" / "detour_expert_20260502_172927_summary.json",
        "bc_checkpoint": ROOT / "artifacts" / "checkpoints" / "bc" / "bc_20260502_172938" / "checkpoint.pt",
        "bc_eval": ROOT / "artifacts" / "evals" / "demo_qty" / "q50" / "bc_eval_20260503_235133.json",
    },
    200: {
        "dataset_npz": ROOT / "artifacts" / "datasets" / "demo_qty" / "q200" / "detour_expert_20260503_235013.npz",
        "dataset_summary": ROOT / "artifacts" / "datasets" / "demo_qty" / "q200" / "detour_expert_20260503_235013_summary.json",
        "bc_checkpoint": ROOT / "artifacts" / "checkpoints" / "bc_demo_qty" / "q200" / "bc_20260503_235025" / "checkpoint.pt",
        "bc_eval": ROOT / "artifacts" / "evals" / "demo_qty" / "q200" / "bc_eval_20260503_235123.json",
    },
}


def collect_eval_start_goal_pairs(episodes: int = 50, seed: int = 300):
    env = DetourWaypointVelocityAviary(gui=False)
    start_goal = []
    for episode_idx in range(episodes):
        _, info = env.reset(seed=seed + episode_idx)
        start = np.asarray(info["position"], dtype=np.float32)
        goal = np.asarray(info["goal"], dtype=np.float32)
        start_goal.append(np.concatenate([start, goal], axis=0))
    env.close()
    return np.stack(start_goal)


def nearest_neighbor_distance(eval_pairs: np.ndarray, demo_pairs: np.ndarray) -> dict[str, float]:
    # Normalize by the eval-set standard deviation so x/y/z scales do not dominate unevenly.
    scale = eval_pairs.std(axis=0) + 1e-6
    normalized_eval = eval_pairs / scale
    normalized_demo = demo_pairs / scale
    dists = np.linalg.norm(
        normalized_eval[:, None, :] - normalized_demo[None, :, :],
        axis=-1,
    )
    min_dists = dists.min(axis=1)
    return {
        "mean_nn_distance": float(min_dists.mean()),
        "p90_nn_distance": float(np.percentile(min_dists, 90)),
        "max_nn_distance": float(min_dists.max()),
    }


def classify_bc_rollout(checkpoint_path: Path, episodes: int = 50, seed: int = 300):
    device = torch.device("cpu")
    model, obs_mean, obs_std, _ = load_bc_checkpoint(checkpoint_path, device)
    env = DetourWaypointVelocityAviary(gui=False)

    categories: list[str] = []
    stage_rank = {"entry": 0, "exit": 1, "goal": 2}

    for episode_idx in range(episodes):
        obs, info = env.reset(seed=seed + episode_idx)
        pos = np.asarray(info["position"], dtype=np.float32)
        start_stage, _ = env._detour_navigation_target(pos)
        max_rank = stage_rank[start_stage]
        final_info = info

        while True:
            action = predict_action(model, obs, obs_mean, obs_std, device)
            obs, _, terminated, truncated, final_info = env.step(action)
            stage = str(final_info["detour_stage"])
            max_rank = max(max_rank, stage_rank[stage])
            if terminated or truncated:
                break

        if bool(final_info["success"]):
            categories.append("success")
        elif max_rank >= stage_rank["goal"]:
            categories.append("reached_goal_stage_failed_to_settle")
        elif max_rank >= stage_rank["exit"]:
            categories.append("reached_exit_only")
        else:
            categories.append("stuck_before_exit")

    env.close()
    counts = Counter(categories)
    ordered = [
        "stuck_before_exit",
        "reached_exit_only",
        "reached_goal_stage_failed_to_settle",
        "success",
    ]
    return {key: int(counts.get(key, 0)) for key in ordered}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_pairs = collect_eval_start_goal_pairs()
    summary: dict[int, dict[str, object]] = {}

    for quantity, paths in CONFIG.items():
        data = np.load(paths["dataset_npz"])
        starts = data["episode_initial_positions"].astype(np.float32)
        goals = data["episode_goals"].astype(np.float32)
        demo_pairs = np.concatenate([starts, goals], axis=1)

        dataset_summary = json.loads(paths["dataset_summary"].read_text(encoding="utf-8"))
        bc_eval = json.loads(paths["bc_eval"].read_text(encoding="utf-8"))
        coverage = nearest_neighbor_distance(eval_pairs, demo_pairs)
        stage_counts = classify_bc_rollout(paths["bc_checkpoint"])

        summary[quantity] = {
            "episodes": int(dataset_summary["episodes"]),
            "expert_success_rate": float(dataset_summary["success_rate"]),
            "expert_mean_final_distance": float(dataset_summary["mean_final_distance"]),
            "bc_only_success_rate": float(bc_eval["success_rate"]),
            "bc_only_mean_final_distance": float(bc_eval["mean_final_distance"]),
            "coverage": coverage,
            "bc_failure_stage_counts": stage_counts,
            "starts": starts.tolist(),
            "goals": goals.tolist(),
        }

    summary_path = OUT_DIR / "coverage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    quantities = [10, 50, 200]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))

    # Top-down coverage scatter.
    ax = axes[0, 0]
    colors = {10: "#2563eb", 50: "#7c3aed", 200: "#059669"}
    for quantity in quantities:
        starts = np.asarray(summary[quantity]["starts"], dtype=np.float32)
        goals = np.asarray(summary[quantity]["goals"], dtype=np.float32)
        ax.scatter(starts[:, 0], starts[:, 1], s=14, alpha=0.55, color=colors[quantity], label=f"{quantity} demos start")
        ax.scatter(goals[:, 0], goals[:, 1], s=14, alpha=0.35, color=colors[quantity], marker="x", label=f"{quantity} demos goal")
    ax.axvline(0.0, color="#b91c1c", linestyle="--", linewidth=1)
    ax.set_title("Start/Goal Coverage (Top-down XY)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    # Nearest-neighbor coverage.
    ax = axes[0, 1]
    nn_means = [summary[q]["coverage"]["mean_nn_distance"] for q in quantities]
    nn_p90 = [summary[q]["coverage"]["p90_nn_distance"] for q in quantities]
    width = 0.34
    x = np.arange(len(quantities))
    ax.bar(x - width / 2, nn_means, width=width, label="Mean NN distance")
    ax.bar(x + width / 2, nn_p90, width=width, label="P90 NN distance")
    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in quantities])
    ax.set_xlabel("Demo episodes")
    ax.set_ylabel("Normalized distance")
    ax.set_title("Eval Start/Goal Coverage")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)

    # BC-only stage failure analysis.
    ax = axes[1, 0]
    categories = [
        "stuck_before_exit",
        "reached_exit_only",
        "reached_goal_stage_failed_to_settle",
        "success",
    ]
    bottom = np.zeros(len(quantities))
    palette = {
        "stuck_before_exit": "#ef4444",
        "reached_exit_only": "#f59e0b",
        "reached_goal_stage_failed_to_settle": "#3b82f6",
        "success": "#10b981",
    }
    labels = {
        "stuck_before_exit": "Stuck before exit",
        "reached_exit_only": "Reached exit only",
        "reached_goal_stage_failed_to_settle": "Reached goal stage, failed settle",
        "success": "Success",
    }
    for category in categories:
        values = np.array([summary[q]["bc_failure_stage_counts"][category] for q in quantities], dtype=float)
        values /= 50.0
        ax.bar(np.arange(len(quantities)), values, bottom=bottom, color=palette[category], label=labels[category])
        bottom += values
    ax.set_xticks(np.arange(len(quantities)))
    ax.set_xticklabels([str(q) for q in quantities])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Demo episodes")
    ax.set_ylabel("Fraction of BC-only eval episodes")
    ax.set_title("BC-only Failure Stage")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=7)

    # Text panel with compact takeaways.
    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        "Coverage / quality takeaways",
        "",
        f"10 demos: expert quality already good ({summary[10]['expert_success_rate']:.2f} success),",
        f"but coverage sparse (mean NN {summary[10]['coverage']['mean_nn_distance']:.2f})",
        f"and BC-only mostly fails before or near exit.",
        "",
        f"50 demos: BC-only still weak ({summary[50]['bc_only_success_rate']:.2f}),",
        "but coverage is much better and this was the strongest BC+PPO regime.",
        "",
        f"200 demos: BC-only already solves the task ({summary[200]['bc_only_success_rate']:.2f}),",
        "so poor BC+PPO results here are less likely to be a coverage problem",
        "and more likely due to PPO disturbing an already-strong BC policy.",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "coverage_overview.png", dpi=180, bbox_inches="tight")

    print(f"Saved coverage summary to {summary_path}")
    print(f"Saved figure to {OUT_DIR / 'coverage_overview.png'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
