from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final portfolio comparison tables and figures.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis/portfolio_final"))
    parser.add_argument(
        "--bc-eval",
        type=Path,
        default=Path("artifacts/evals/bc_clean50_50eps_seed20000_20260609.json"),
    )
    parser.add_argument(
        "--strong-reg-50ep",
        type=Path,
        default=Path("artifacts/evals/strong_reg_best_50eps_20260609/best_50eps_summary.csv"),
    )
    parser.add_argument(
        "--aug005-evals",
        type=Path,
        nargs="+",
        default=[
            Path("artifacts/evals/aug005_final_seed7_50eps_seed20000_20260609.json"),
            Path("artifacts/evals/aug005_final_seed19_50eps_seed20000_20260609.json"),
            Path("artifacts/evals/aug005_more_final_seed11_50eps_seed20000_20260609.json"),
            Path("artifacts/evals/aug005_more_final_seed23_50eps_seed20000_20260609.json"),
            Path("artifacts/evals/aug005_more_final_seed31_50eps_seed20000_20260609.json"),
        ],
    )
    parser.add_argument(
        "--aug005-best-evals",
        type=Path,
        nargs="+",
        default=[
            Path("artifacts/evals/aug005_best_seed7_50eps_seed20000_20260609.json"),
            Path("artifacts/evals/aug005_best_seed11_50eps_seed20000_20260609.json"),
            Path("artifacts/evals/aug005_best_seed19_50eps_seed20000_20260609.json"),
            Path("artifacts/evals/aug005_best_seed23_50eps_seed20000_20260609.json"),
            Path("artifacts/evals/aug005_best_seed31_50eps_seed20000_20260609.json"),
        ],
    )
    parser.add_argument(
        "--strong-reg-summary",
        type=Path,
        default=Path("artifacts/analysis/ppo_bc_strong_regularization_ablation_metrics_rerun_20260609_092402/variant_summary.csv"),
    )
    parser.add_argument(
        "--expert-bc-summary",
        type=Path,
        default=Path("artifacts/analysis/ppo_expert_bc_loss_ablation_20260609_112015/variant_summary.csv"),
    )
    parser.add_argument(
        "--aug005-roots",
        type=Path,
        nargs="+",
        default=[
            Path("artifacts/checkpoints/ppo_expert_bc_aug_ablation_20260609_124357"),
            Path("artifacts/checkpoints/ppo_expert_bc_aug005_more_seeds_20260609_135343"),
        ],
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float32)
    return float(array.mean()), float(array.std(ddof=0))


def build_headline_50ep(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bc_eval = load_json(args.bc_eval)
    rows.append(
        {
            "method": "BC-only",
            "protocol": "50 episodes, seed 20000",
            "num_policy_seeds": 1,
            "success_rate_mean": bc_eval["success_rate"],
            "success_rate_std": 0.0,
            "mean_final_distance_mean": bc_eval["mean_final_distance"],
            "mean_final_distance_std": 0.0,
            "mean_episode_return_mean": bc_eval["mean_episode_return"],
            "mean_episode_return_std": 0.0,
            "source": str(args.bc_eval),
        }
    )

    strong_reg = pd.read_csv(args.strong_reg_50ep)
    stable = strong_reg[strong_reg["variant"] == "low_std_kl3e3_freeze50k"]
    rows.append(
        {
            "method": "BC+PPO KL/freeze best",
            "protocol": "50 episodes, seed 20000",
            "num_policy_seeds": int(len(stable)),
            "success_rate_mean": float(stable["success_rate"].mean()),
            "success_rate_std": float(stable["success_rate"].std(ddof=0)),
            "mean_final_distance_mean": float(stable["mean_final_distance"].mean()),
            "mean_final_distance_std": float(stable["mean_final_distance"].std(ddof=0)),
            "mean_episode_return_mean": float(stable["mean_episode_return"].mean()),
            "mean_episode_return_std": float(stable["mean_episode_return"].std(ddof=0)),
            "source": str(args.strong_reg_50ep),
        }
    )

    rows.append(build_eval_set_row("BC+PPO expert-state aug final", args.aug005_evals))
    rows.append(build_eval_set_row("BC+PPO expert-state aug best", args.aug005_best_evals))
    return pd.DataFrame(rows)


def build_eval_set_row(method: str, paths: list[Path]) -> dict[str, Any]:
    eval_rows = [load_json(path) for path in paths]
    success_mean, success_std = mean_std([float(row["success_rate"]) for row in eval_rows])
    distance_mean, distance_std = mean_std([float(row["mean_final_distance"]) for row in eval_rows])
    return_mean, return_std = mean_std([float(row["mean_episode_return"]) for row in eval_rows])
    return {
        "method": method,
        "protocol": "50 episodes, seed 20000",
        "num_policy_seeds": len(eval_rows),
        "success_rate_mean": success_mean,
        "success_rate_std": success_std,
        "mean_final_distance_mean": distance_mean,
        "mean_final_distance_std": distance_std,
        "mean_episode_return_mean": return_mean,
        "mean_episode_return_std": return_std,
        "source": ";".join(str(path) for path in paths),
    }


def final_records_from_roots(roots: list[Path]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for root in roots:
        for summary_path in sorted(root.glob("**/summary.json")):
            summary = load_json(summary_path)
            final = summary.get("final_eval")
            if not final:
                continue
            records.append(
                {
                    "seed": int(summary["seed"]),
                    "success_rate": float(final["success_rate"]),
                    "mean_final_distance": float(final["mean_final_distance"]),
                    "collision_rate": float(final.get("collision_rate", np.nan)),
                    "bc_probe_action_l2": float(final.get("bc_probe_action_l2", np.nan)),
                    "source": str(summary_path),
                }
            )
    return pd.DataFrame(records)


def build_online_diagnostics(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    strong = pd.read_csv(args.strong_reg_summary)
    strong_row = strong[
        (strong["variant"] == "low_std_kl3e3_freeze50k") & (strong["checkpoint"] == "final")
    ].iloc[0]
    rows.append(
        {
            "method": "BC+PPO KL/freeze",
            "protocol": "10 episode online eval",
            "num_policy_seeds": int(strong_row["num_runs"]),
            "success_rate_mean": float(strong_row["success_rate_mean"]),
            "success_rate_std": float(strong_row["success_rate_std"]),
            "mean_final_distance_mean": float(strong_row["mean_final_distance_mean"]),
            "collision_rate_mean": float(strong_row["collision_rate_mean"]),
            "bc_probe_action_l2_mean": float(strong_row["bc_probe_action_l2_mean"]),
            "source": str(args.strong_reg_summary),
        }
    )

    expert = pd.read_csv(args.expert_bc_summary)
    expert_row = expert[
        (expert["variant"] == "expertbc_1p0_kl3e3_freeze25k") & (expert["checkpoint"] == "final")
    ].iloc[0]
    rows.append(
        {
            "method": "BC+PPO expert BC loss",
            "protocol": "10 episode online eval",
            "num_policy_seeds": int(expert_row["num_runs"]),
            "success_rate_mean": float(expert_row["success_rate_mean"]),
            "success_rate_std": float(expert_row["success_rate_std"]),
            "mean_final_distance_mean": float(expert_row["mean_final_distance_mean"]),
            "collision_rate_mean": float(expert_row["collision_rate_mean"]),
            "bc_probe_action_l2_mean": float(expert_row["bc_probe_action_l2_mean"]),
            "source": str(args.expert_bc_summary),
        }
    )

    aug = final_records_from_roots(args.aug005_roots)
    rows.append(
        {
            "method": "BC+PPO expert-state aug",
            "protocol": "10 episode online eval",
            "num_policy_seeds": int(len(aug)),
            "success_rate_mean": float(aug["success_rate"].mean()),
            "success_rate_std": float(aug["success_rate"].std(ddof=0)),
            "mean_final_distance_mean": float(aug["mean_final_distance"].mean()),
            "collision_rate_mean": float(aug["collision_rate"].mean()),
            "bc_probe_action_l2_mean": float(aug["bc_probe_action_l2"].mean()),
            "source": ";".join(str(root) for root in args.aug005_roots),
        }
    )
    return pd.DataFrame(rows)


def plot_bar_comparison(df: pd.DataFrame, output_dir: Path, prefix: str, metrics: list[tuple[str, str]]) -> None:
    for metric, ylabel in metrics:
        plt.figure(figsize=(8, 4.8))
        xs = np.arange(len(df))
        means = df[f"{metric}_mean"].to_numpy(dtype=np.float32)
        std_col = f"{metric}_std"
        yerr = df[std_col].to_numpy(dtype=np.float32) if std_col in df.columns else None
        colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
        plt.bar(xs, means, yerr=yerr, capsize=4, color=colors[: len(df)])
        plt.xticks(xs, df["method"], rotation=15, ha="right")
        plt.ylabel(ylabel)
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_{metric}.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    headline = build_headline_50ep(args)
    diagnostics = build_online_diagnostics(args)

    headline.to_csv(args.output_dir / "headline_50ep_comparison.csv", index=False)
    diagnostics.to_csv(args.output_dir / "online_diagnostics_comparison.csv", index=False)

    plot_bar_comparison(
        headline,
        args.output_dir,
        "headline_50ep",
        [
            ("success_rate", "Success rate"),
            ("mean_final_distance", "Mean final distance (m)"),
            ("mean_episode_return", "Mean episode return"),
        ],
    )
    plot_bar_comparison(
        diagnostics,
        args.output_dir,
        "online_diagnostics",
        [
            ("success_rate", "Success rate"),
            ("collision_rate", "Collision rate"),
            ("bc_probe_action_l2", "BC probe action L2"),
            ("mean_final_distance", "Mean final distance (m)"),
        ],
    )

    manifest = {
        "output_dir": str(args.output_dir),
        "headline_rows": headline.to_dict(orient="records"),
        "diagnostic_rows": diagnostics.to_dict(orient="records"),
        "outputs": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
