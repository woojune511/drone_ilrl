from __future__ import annotations

import importlib
import importlib.metadata as metadata
import platform
import sys


REQUIRED_PACKAGES = [
    ("gymnasium", "gymnasium"),
    ("stable_baselines3", "stable-baselines3"),
    ("gym_pybullet_drones", "gym-pybullet-drones"),
]

OPTIONAL_PACKAGES = [
    ("imitation", "imitation"),
]


def package_version(distribution_name: str) -> str:
    return metadata.version(distribution_name)


def check_import(import_name: str) -> None:
    importlib.import_module(import_name)


def detect_pybullet_distribution() -> tuple[str, str]:
    check_import("pybullet")
    for dist_name in ("pybullet", "pybullet-arm64"):
        try:
            return "pybullet", f"{dist_name} ({package_version(dist_name)})"
        except metadata.PackageNotFoundError:
            continue
    return "pybullet", "unknown distribution"


def main() -> None:
    print("=== Environment Check ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version:    {platform.python_version()}")
    print(f"Platform:          {platform.platform()}")
    print()

    print("Required packages")
    for import_name, dist_name in REQUIRED_PACKAGES:
        check_import(import_name)
        print(f"- {dist_name}: {package_version(dist_name)}")
    pybullet_import, pybullet_dist = detect_pybullet_distribution()
    print(f"- {pybullet_import}: {pybullet_dist}")

    print()
    print("Optional packages")
    for import_name, dist_name in OPTIONAL_PACKAGES:
        try:
            check_import(import_name)
            print(f"- {dist_name}: {package_version(dist_name)}")
        except ModuleNotFoundError:
            print(f"- {dist_name}: not installed")

    print()
    print("Environment looks ready for IL/RL experiments.")


if __name__ == "__main__":
    main()
