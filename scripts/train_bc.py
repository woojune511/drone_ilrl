from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ilrl_lab.bc import BehaviorCloningPolicy, normalize_obs, save_bc_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a behavior-cloning policy on expert rollouts.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to an expert dataset (.npz). Defaults to the latest file in artifacts/datasets.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("artifacts") / "datasets",
        help="Directory used when --dataset is omitted.",
    )
    parser.add_argument(
        "--dataset-glob",
        type=str,
        default="*_expert_*.npz",
        help="Glob pattern used inside --dataset-dir when --dataset is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "checkpoints" / "bc",
        help="Directory where checkpoints and metrics are saved.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    return parser.parse_args()


def latest_dataset(dataset_dir: Path, dataset_glob: str) -> Path:
    candidates = sorted(
        [path for path in dataset_dir.glob(dataset_glob) if not path.name.endswith("_summary.npz")],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No expert dataset found under {dataset_dir} matching {dataset_glob}")
    return candidates[-1]


def split_by_episode(
    episode_ids: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_ids = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)
    val_count = max(1, int(len(unique_ids) * val_fraction))
    val_ids = set(int(x) for x in shuffled[:val_count])
    train_mask = np.array([int(ep) not in val_ids for ep in episode_ids], dtype=bool)
    val_mask = ~train_mask
    return train_mask, val_mask


def build_loader(obs: np.ndarray, acts: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(obs.astype(np.float32)),
        torch.from_numpy(acts.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: BehaviorCloningPolicy, loader: DataLoader, loss_fn: nn.Module, device: torch.device):
    model.eval()
    losses: list[float] = []
    maes: list[float] = []
    with torch.no_grad():
        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            pred = model(obs_batch)
            loss = loss_fn(pred, act_batch)
            mae = torch.mean(torch.abs(pred - act_batch))
            losses.append(float(loss.item()))
            maes.append(float(mae.item()))
    return float(np.mean(losses)), float(np.mean(maes))


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset if args.dataset is not None else latest_dataset(args.dataset_dir, args.dataset_glob)
    data = np.load(dataset_path)

    obs = data["obs"].astype(np.float32)
    acts = data["acts"].astype(np.float32)
    episode_ids = data["episode_ids"].astype(np.int32)

    train_mask, val_mask = split_by_episode(episode_ids, args.val_fraction, args.seed)
    train_obs_raw = obs[train_mask]
    train_acts = acts[train_mask]
    val_obs_raw = obs[val_mask]
    val_acts = acts[val_mask]

    obs_mean = train_obs_raw.mean(axis=0)
    obs_std = train_obs_raw.std(axis=0) + 1e-6

    train_obs = normalize_obs(train_obs_raw, obs_mean, obs_std)
    val_obs = normalize_obs(val_obs_raw, obs_mean, obs_std)

    train_loader = build_loader(train_obs, train_acts, args.batch_size, shuffle=True)
    val_loader = build_loader(val_obs, val_acts, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BehaviorCloningPolicy(
        obs_dim=int(obs.shape[1]),
        action_dim=int(acts.shape[1]),
        hidden_sizes=tuple(args.hidden_sizes),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")

    run_name = datetime.now().strftime("bc_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    checkpoint_path = run_dir / "checkpoint.pt"
    metrics_path = run_dir / "metrics.json"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(obs_batch)
            loss = loss_fn(pred, act_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        val_loss, val_mae = evaluate(model, val_loader, loss_fn, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
        }
        history.append(record)
        print(json.dumps(record))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            metadata = {
                "dataset_path": str(dataset_path),
                "obs_dim": int(obs.shape[1]),
                "action_dim": int(acts.shape[1]),
                "hidden_sizes": list(args.hidden_sizes),
                "best_val_loss": best_val_loss,
            }
            save_bc_checkpoint(checkpoint_path, model, obs_mean, obs_std, metadata)

    summary = {
        "dataset_path": str(dataset_path),
        "dataset_glob": args.dataset_glob,
        "train_transitions": int(train_mask.sum()),
        "val_transitions": int(val_mask.sum()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "best_val_loss": best_val_loss,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
