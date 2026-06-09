from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


EVAL_METRICS = [
    "success_rate",
    "mean_final_distance",
    "mean_min_distance",
    "position_only_success_rate",
    "collision_rate",
    "reached_exit_stage_rate",
    "reached_goal_stage_rate",
    "mean_final_speed",
    "mean_episode_return",
    "mean_episode_length",
    "policy_std_mean",
    "policy_log_std_mean",
    "bc_probe_action_l2",
    "bc_probe_action_cosine",
    "bc_probe_max_abs_action_diff",
    "bc_probe_action_saturation_rate",
]

TB_TAGS = [
    "train/approx_kl",
    "train/entropy_loss",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/loss",
    "train/explained_variance",
    "train/clip_fraction",
    "train/bc_kl_loss",
    "train/bc_kl_loss_weighted",
    "train/expert_bc_loss",
    "train/expert_bc_loss_weighted",
    "train/std",
    "train/log_std_mean",
    "train/log_std_min",
    "train/log_std_max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training/evaluation diagnostics from PPO runs.")
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        required=True,
        help="Root directories containing run summary.json files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--total-timesteps-filter",
        type=int,
        default=None,
        help="Only include runs with this total_timesteps value.",
    )
    return parser.parse_args()


def discover_summaries(roots: list[Path], total_timesteps_filter: int | None) -> list[tuple[Path, dict[str, Any]]]:
    summaries: list[tuple[Path, dict[str, Any]]] = []
    for root in roots:
        for path in sorted(root.glob("**/summary.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if total_timesteps_filter is not None and payload.get("total_timesteps") != total_timesteps_filter:
                continue
            payload["_summary_path"] = str(path)
            payload["_variant"] = infer_variant(path)
            summaries.append((path, payload))
    if not summaries:
        raise FileNotFoundError("No matching summary.json files found.")
    return summaries


def infer_variant(summary_path: Path) -> str:
    # Expected: root / variant / task / run / summary.json.
    if len(summary_path.parents) >= 3:
        return summary_path.parents[2].name
    return "run"


def load_eval_rows(summaries: list[tuple[Path, dict[str, Any]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, summary in summaries:
        history_path = Path(summary["eval_history_path"])
        if not history_path.exists():
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        for record in history:
            row = {
                "variant": summary["_variant"],
                "seed": int(summary["seed"]),
                "run_dir": summary["run_dir"],
                "timesteps": int(record["timesteps"]),
            }
            for key in EVAL_METRICS:
                if record.get(key) is not None:
                    row[key] = float(record[key])
            rows.append(row)
    return pd.DataFrame(rows)


def load_best_final_rows(summaries: list[tuple[Path, dict[str, Any]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, summary in summaries:
        for label, record in (("best", summary.get("best_eval")), ("final", summary.get("final_eval"))):
            if not record:
                continue
            row = {
                "variant": summary["_variant"],
                "seed": int(summary["seed"]),
                "checkpoint": label,
                "run_dir": summary["run_dir"],
            }
            for key in EVAL_METRICS:
                if record.get(key) is not None:
                    row[key] = float(record[key])
            rows.append(row)
    return pd.DataFrame(rows)


def load_tensorboard_rows(summaries: list[tuple[Path, dict[str, Any]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, summary in summaries:
        tb_root = Path(summary["run_dir"]) / "tensorboard"
        if not tb_root.exists():
            continue
        for event_path in tb_root.glob("**/events.out.tfevents.*"):
            accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
            try:
                accumulator.Reload()
            except Exception:
                continue
            available_tags = set(accumulator.Tags().get("scalars", []))
            for tag in TB_TAGS:
                if tag not in available_tags:
                    continue
                for event in accumulator.Scalars(tag):
                    rows.append(
                        {
                            "variant": summary["_variant"],
                            "seed": int(summary["seed"]),
                            "run_dir": summary["run_dir"],
                            "tag": tag,
                            "step": int(event.step),
                            "value": float(event.value),
                        }
                    )
    return pd.DataFrame(rows)


def plot_eval_metric(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    if metric not in df.columns:
        return
    metric_df = df.dropna(subset=[metric])
    if metric_df.empty:
        return
    plt.figure(figsize=(8, 5))
    for variant, group in metric_df.groupby("variant"):
        grouped = group.groupby("timesteps")[metric]
        xs = np.asarray(sorted(grouped.groups.keys()), dtype=np.float32)
        means = np.asarray([grouped.get_group(step).mean() for step in xs], dtype=np.float32)
        stds = np.asarray([grouped.get_group(step).std(ddof=0) for step in xs], dtype=np.float32)
        plt.plot(xs, means, marker="o", label=variant)
        plt.fill_between(xs, means - stds, means + stds, alpha=0.15)
    plt.xlabel("Environment Steps")
    plt.ylabel(metric)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_tb_tag(df: pd.DataFrame, tag: str, output_path: Path) -> None:
    tag_df = df[df["tag"] == tag]
    if tag_df.empty:
        return
    plt.figure(figsize=(8, 5))
    for variant, group in tag_df.groupby("variant"):
        grouped = group.groupby("step")["value"]
        xs = np.asarray(sorted(grouped.groups.keys()), dtype=np.float32)
        means = np.asarray([grouped.get_group(step).mean() for step in xs], dtype=np.float32)
        stds = np.asarray([grouped.get_group(step).std(ddof=0) for step in xs], dtype=np.float32)
        plt.plot(xs, means, label=variant)
        plt.fill_between(xs, means - stds, means + stds, alpha=0.15)
    plt.xlabel("Environment Steps")
    plt.ylabel(tag)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_variant_summary(best_final_df: pd.DataFrame, output_dir: Path) -> None:
    if best_final_df.empty:
        return
    metric_keys = [key for key in EVAL_METRICS if key in best_final_df.columns]
    rows: list[dict[str, Any]] = []
    for (variant, checkpoint), group in best_final_df.groupby(["variant", "checkpoint"]):
        row: dict[str, Any] = {
            "variant": variant,
            "checkpoint": checkpoint,
            "num_runs": int(len(group)),
        }
        for key in metric_keys:
            values = group[key].dropna()
            if values.empty:
                continue
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_std"] = float(values.std(ddof=0))
        rows.append(row)
    summary_df = pd.DataFrame(rows).sort_values(["variant", "checkpoint"])
    summary_df.to_csv(output_dir / "variant_summary.csv", index=False)
    (output_dir / "variant_summary.json").write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = discover_summaries(args.roots, args.total_timesteps_filter)
    eval_df = load_eval_rows(summaries)
    best_final_df = load_best_final_rows(summaries)
    tb_df = load_tensorboard_rows(summaries)

    if not eval_df.empty:
        eval_df.to_csv(args.output_dir / "eval_diagnostics_long.csv", index=False)
        for metric in EVAL_METRICS:
            plot_eval_metric(eval_df, metric, args.output_dir / f"eval_{metric}.png")

    if not best_final_df.empty:
        best_final_df.to_csv(args.output_dir / "best_final_diagnostics.csv", index=False)
        write_variant_summary(best_final_df, args.output_dir)

    if not tb_df.empty:
        tb_df.to_csv(args.output_dir / "tensorboard_scalars_long.csv", index=False)
        for tag in TB_TAGS:
            plot_tb_tag(tb_df, tag, args.output_dir / f"tb_{tag.replace('/', '_')}.png")

    manifest = {
        "num_runs": len(summaries),
        "roots": [str(root) for root in args.roots],
        "eval_metrics": EVAL_METRICS,
        "tensorboard_tags": TB_TAGS,
        "outputs": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
