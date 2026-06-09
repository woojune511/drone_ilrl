from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from ilrl_lab.bc import load_bc_checkpoint
from ilrl_lab.ppo_training import (
    PeriodicEvalCallback,
    build_bc_probe,
    build_ppo_model,
    build_training_env,
    evaluate_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO baseline on the waypoint environment.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "checkpoints" / "ppo_scratch",
        help="Directory where PPO runs and evaluation logs are saved.",
    )
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel training environments. Use >1 to speed up PyBullet rollouts.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument(
        "--log-std-init",
        type=float,
        default=0.0,
        help="Initial log standard deviation for the PPO Gaussian policy.",
    )
    parser.add_argument(
        "--obs-norm-bc-checkpoint",
        type=Path,
        default=None,
        help="Optional BC checkpoint whose obs_mean/obs_std will be used to normalize PPO observations.",
    )
    parser.add_argument("--gui", action="store_true", help="Enable GUI for the training environment.")
    parser.add_argument(
        "--task-variant",
        choices=["waypoint", "detour"],
        default="waypoint",
        help="Environment variant to train on.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = datetime.now().strftime(f"ppo_seed{args.seed}_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / args.task_variant / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = run_dir

    obs_mean = None
    obs_std = None
    bc_probe = None
    if args.obs_norm_bc_checkpoint is not None:
        _, obs_mean, obs_std, _ = load_bc_checkpoint(args.obs_norm_bc_checkpoint, torch.device("cpu"))
        bc_probe = build_bc_probe(
            task_variant=args.task_variant,
            bc_checkpoint=args.obs_norm_bc_checkpoint,
            obs_mean=obs_mean,
            obs_std=obs_std,
            seed=args.seed + 30_000,
        )

    env = build_training_env(
        gui=args.gui,
        seed=args.seed,
        task_variant=args.task_variant,
        obs_mean=obs_mean,
        obs_std=obs_std,
        n_envs=args.n_envs,
    )
    model = build_ppo_model(env, args.seed, args)

    callback = PeriodicEvalCallback(
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        eval_seed=args.seed + 10_000,
        run_dir=run_dir,
        task_variant=args.task_variant,
        obs_mean=obs_mean,
        obs_std=obs_std,
        bc_probe=bc_probe,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=False)
    final_model_path = run_dir / "final_model.zip"
    model.save(final_model_path)

    final_eval, _ = evaluate_model(
        model,
        args.eval_episodes,
        args.seed + 20_000,
        task_variant=args.task_variant,
        obs_mean=obs_mean,
        obs_std=obs_std,
        bc_probe=bc_probe,
    )
    final_eval.timesteps = int(args.total_timesteps)

    summary = {
        "run_dir": str(run_dir),
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_envs": args.n_envs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "log_std_init": args.log_std_init,
        "obs_norm_bc_checkpoint": None if args.obs_norm_bc_checkpoint is None else str(args.obs_norm_bc_checkpoint),
        "bc_probe": None
        if bc_probe is None
        else {
            "num_states": bc_probe["num_states"],
            "seed": bc_probe["seed"],
            "episodes": bc_probe["episodes"],
            "max_steps_per_episode": bc_probe["max_steps_per_episode"],
            "stride": bc_probe["stride"],
        },
        "task_variant": args.task_variant,
        "final_model_path": str(final_model_path),
        "best_model_path": str(callback.best_model_path),
        "best_eval_path": str(callback.best_record_path),
        "best_eval": None if callback.best_record is None else asdict(callback.best_record),
        "best_model_selection": "highest_success_rate_then_lowest_final_distance_then_highest_return",
        "eval_history_path": str(callback.history_path),
        "final_eval": asdict(final_eval),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved PPO run to {run_dir}")
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
