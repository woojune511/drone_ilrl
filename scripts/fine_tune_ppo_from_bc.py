from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from ilrl_lab.ppo_training import (
    PeriodicEvalCallback,
    build_ppo_model,
    build_training_env,
    evaluate_model,
    initialize_actor_from_bc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune PPO from a BC-initialized actor.")
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        required=True,
        help="Path to the BC checkpoint used to initialize the PPO actor.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "checkpoints" / "ppo_bc_init",
        help="Directory where PPO runs and evaluation logs are saved.",
    )
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--gui", action="store_true", help="Enable GUI for the training environment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = datetime.now().strftime("ppo_bc_init_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = run_dir

    env = build_training_env(gui=args.gui, seed=args.seed)
    model = build_ppo_model(env, args.seed, args)
    init_info = initialize_actor_from_bc(model, args.bc_checkpoint, torch.device("cpu"))

    callback = PeriodicEvalCallback(
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        eval_seed=args.seed + 10_000,
        run_dir=run_dir,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=False)
    final_model_path = run_dir / "final_model.zip"
    model.save(final_model_path)

    final_eval, _ = evaluate_model(model, args.eval_episodes, args.seed + 20_000)
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
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "bc_checkpoint": str(args.bc_checkpoint),
        "initialization": init_info,
        "final_model_path": str(final_model_path),
        "best_model_path": str(callback.best_model_path),
        "eval_history_path": str(callback.history_path),
        "final_eval": asdict(final_eval),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved BC-initialized PPO run to {run_dir}")
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
