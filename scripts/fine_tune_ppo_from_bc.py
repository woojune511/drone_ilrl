from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ilrl_lab.bc import load_bc_checkpoint
from ilrl_lab.ppo_training import (
    build_bc_probe,
    build_bc_fine_tune_callback,
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
        "--bc-kl-coef",
        type=float,
        default=0.0,
        help="Coefficient for KL-style regularization toward the BC policy mean.",
    )
    parser.add_argument(
        "--freeze-actor-steps",
        type=int,
        default=0,
        help="Number of initial environment steps to freeze the BC-initialized actor layers.",
    )
    parser.add_argument(
        "--freeze-actor-mode",
        choices=["all", "hidden_only"],
        default="all",
        help="Which BC-initialized actor layers to freeze during the warm-start period.",
    )
    parser.add_argument(
        "--expert-bc-loss-coef",
        type=float,
        default=0.0,
        help="Coefficient for supervised MSE loss on expert dataset states/actions during PPO updates.",
    )
    parser.add_argument(
        "--expert-bc-loss-batch-size",
        type=int,
        default=256,
        help="Expert minibatch size used for --expert-bc-loss-coef.",
    )
    parser.add_argument(
        "--expert-bc-dataset",
        type=Path,
        default=None,
        help="Optional expert dataset .npz. Defaults to the BC checkpoint metadata dataset_path.",
    )
    parser.add_argument(
        "--expert-bc-augment-copies",
        type=int,
        default=0,
        help="Number of relabeled noisy copies to add to the expert BC loss dataset.",
    )
    parser.add_argument(
        "--expert-bc-position-noise-std",
        type=float,
        default=0.0,
        help="Gaussian position noise std for expert BC state augmentation.",
    )
    parser.add_argument(
        "--expert-bc-velocity-noise-std",
        type=float,
        default=0.0,
        help="Gaussian linear velocity noise std for expert BC state augmentation.",
    )
    parser.add_argument(
        "--expert-bc-rpy-noise-std",
        type=float,
        default=0.0,
        help="Gaussian roll/pitch/yaw noise std for expert BC state augmentation.",
    )
    parser.add_argument(
        "--expert-bc-angular-velocity-noise-std",
        type=float,
        default=0.0,
        help="Gaussian angular velocity noise std for expert BC state augmentation.",
    )
    parser.add_argument(
        "--expert-bc-augment-seed",
        type=int,
        default=None,
        help="Seed for expert BC state augmentation. Defaults to seed + 40000.",
    )
    parser.add_argument("--gui", action="store_true", help="Enable GUI for the training environment.")
    parser.add_argument(
        "--task-variant",
        choices=["waypoint", "detour", "detour_planar", "detour_planar_local"],
        default="waypoint",
        help="Environment variant to train on.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = datetime.now().strftime(f"ppo_bc_init_seed{args.seed}_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / args.task_variant / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = run_dir

    _, obs_mean, obs_std, bc_metadata = load_bc_checkpoint(args.bc_checkpoint, torch.device("cpu"))
    bc_probe = build_bc_probe(
        task_variant=args.task_variant,
        bc_checkpoint=args.bc_checkpoint,
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
    init_info = initialize_actor_from_bc(
        model,
        args.bc_checkpoint,
        torch.device("cpu"),
        bc_kl_coef=args.bc_kl_coef,
    )
    expert_bc_dataset_path = args.expert_bc_dataset
    if expert_bc_dataset_path is None and bc_metadata.get("dataset_path") is not None:
        expert_bc_dataset_path = Path(str(bc_metadata["dataset_path"]))
    if args.expert_bc_loss_coef > 0.0:
        if expert_bc_dataset_path is None:
            raise ValueError("--expert-bc-loss-coef requires --expert-bc-dataset or BC metadata dataset_path.")
        expert_data = np.load(expert_bc_dataset_path)
        if not hasattr(model, "set_expert_bc_dataset"):
            raise TypeError("The PPO model does not support expert BC dataset regularization.")
        expert_bc_augment_seed = (
            int(args.expert_bc_augment_seed)
            if args.expert_bc_augment_seed is not None
            else int(args.seed + 40_000)
        )
        model.set_expert_bc_dataset(
            observations=expert_data["obs"],
            actions=expert_data["acts"],
            obs_mean=obs_mean,
            obs_std=obs_std,
            coef=args.expert_bc_loss_coef,
            batch_size=args.expert_bc_loss_batch_size,
            task_variant=args.task_variant,
            augment_copies=args.expert_bc_augment_copies,
            position_noise_std=args.expert_bc_position_noise_std,
            velocity_noise_std=args.expert_bc_velocity_noise_std,
            rpy_noise_std=args.expert_bc_rpy_noise_std,
            angular_velocity_noise_std=args.expert_bc_angular_velocity_noise_std,
            augment_seed=expert_bc_augment_seed,
        )
    else:
        expert_bc_augment_seed = None

    callback, eval_callback = build_bc_fine_tune_callback(
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        eval_seed=args.seed + 10_000,
        run_dir=run_dir,
        task_variant=args.task_variant,
        obs_mean=obs_mean,
        obs_std=obs_std,
        freeze_actor_steps=args.freeze_actor_steps,
        freeze_actor_mode=args.freeze_actor_mode,
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
        "bc_kl_coef": args.bc_kl_coef,
        "expert_bc_loss_coef": args.expert_bc_loss_coef,
        "expert_bc_loss_batch_size": args.expert_bc_loss_batch_size,
        "expert_bc_dataset": None if expert_bc_dataset_path is None else str(expert_bc_dataset_path),
        "expert_bc_augment_copies": args.expert_bc_augment_copies,
        "expert_bc_position_noise_std": args.expert_bc_position_noise_std,
        "expert_bc_velocity_noise_std": args.expert_bc_velocity_noise_std,
        "expert_bc_rpy_noise_std": args.expert_bc_rpy_noise_std,
        "expert_bc_angular_velocity_noise_std": args.expert_bc_angular_velocity_noise_std,
        "expert_bc_augment_seed": expert_bc_augment_seed,
        "expert_bc_num_samples": (
            0
            if not hasattr(model, "expert_bc_observations") or model.expert_bc_observations is None
            else int(model.expert_bc_observations.shape[0])
        ),
        "task_variant": args.task_variant,
        "freeze_actor_steps": args.freeze_actor_steps,
        "freeze_actor_mode": args.freeze_actor_mode,
        "bc_checkpoint": str(args.bc_checkpoint),
        "uses_bc_obs_normalization": True,
        "bc_probe": {
            "num_states": bc_probe["num_states"],
            "seed": bc_probe["seed"],
            "episodes": bc_probe["episodes"],
            "max_steps_per_episode": bc_probe["max_steps_per_episode"],
            "stride": bc_probe["stride"],
        },
        "initialization": init_info,
        "final_model_path": str(final_model_path),
        "best_model_path": str(eval_callback.best_model_path),
        "best_eval_path": str(eval_callback.best_record_path),
        "best_eval": None if eval_callback.best_record is None else asdict(eval_callback.best_record),
        "best_model_selection": "highest_success_rate_then_lowest_final_distance_then_highest_return",
        "eval_history_path": str(eval_callback.history_path),
        "final_eval": asdict(final_eval),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved BC-initialized PPO run to {run_dir}")
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
