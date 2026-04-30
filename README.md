# ILRL Drone Lab

This workspace is set up for rebuilding your old drone imitation-learning project in a lighter stack:

- `uv` for environment and dependency management
- `gym-pybullet-drones` for the drone simulator
- `stable-baselines3` for PPO and related RL baselines
- `imitation` for BC / GAIL style imitation-learning experiments

The drone simulator is wired through a local editable checkout in `vendor/gym-pybullet-drones-main/`, so `uv sync` does not depend on a fragile Git-tag wheel build path.
On Windows, the environment uses `pybullet-arm64` as a drop-in replacement for `pybullet`, because the current `pybullet` release on PyPI does not ship a Windows wheel for this setup and otherwise falls back to a local C++ build.

## Why Python 3.10?

The current `gym-pybullet-drones` project declares `python = "^3.10"` in its project metadata, and the official repository installation example also uses Python 3.10. Pinning to 3.10 keeps the environment conservative and avoids wasting time on package compatibility issues before the experiments even start.

## Quickstart

### 1. Bootstrap the environment

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1
```

This script:

- uses a project-local `.uv-cache` to avoid permission issues with the default global uv cache
- uses a project-local `.uv-python` directory for uv-managed Python installations
- installs Python 3.10 via `uv` if needed
- creates `.venv`
- syncs dependencies for `drones`, `il`, and `dev`
- runs a smoke test

### 2. Activate the virtual environment manually

```powershell
. .\.venv\Scripts\Activate.ps1
```

### 3. Re-run the smoke test

```powershell
uv run python scripts/check_env.py
```

## Manual uv commands

If you want to run the steps manually instead of using the bootstrap script:

```powershell
$env:UV_CACHE_DIR = "$PWD\\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$PWD\\.uv-python"
uv python install 3.10
uv venv --python 3.10
uv sync --extra drones --extra il --extra dev
uv run python scripts/check_env.py
```

## Project Layout

```text
.
├── pyproject.toml
├── README.md
├── scripts/
│   ├── bootstrap.ps1
│   └── check_env.py
├── vendor/
│   └── gym-pybullet-drones-main/
├── src/
│   └── ilrl_lab/
└── tests/
```

## Suggested Next Build Steps

1. Add an expert trajectory collector using the simulator's PID control examples as the starting point.
2. Train a BC baseline and log success rate / trajectory RMSE.
3. Fine-tune the BC checkpoint with PPO and compare convergence speed and final performance.
4. Add a reproducible evaluation script that dumps metrics to `artifacts/`.

## Expert Rollout Collection

The workspace now includes a minimal waypoint task and an automatic expert collector:

```powershell
. .\.venv\Scripts\Activate.ps1
python scripts\collect_expert_rollouts.py --episodes 50
```

This uses:

- `src/ilrl_lab/envs/waypoint_vel_aviary.py` for a single-drone, goal-aware waypoint task
- `src/ilrl_lab/experts/velocity.py` for a simple expert that outputs velocity commands
- the simulator's built-in PID controller to translate high-level velocity commands into motor control

The collector saves:

- compressed transition data to `artifacts/datasets/*.npz`
- a run summary to `artifacts/datasets/*_summary.json`

## Behavior Cloning Baseline

Train the BC policy on the latest collected dataset:

```powershell
. .\.venv\Scripts\Activate.ps1
python scripts\train_bc.py --epochs 15
```

Evaluate the latest BC checkpoint in the environment:

```powershell
python scripts\evaluate_bc.py --episodes 20
```
