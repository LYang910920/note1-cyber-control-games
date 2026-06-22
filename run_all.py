# Copyright (c) 2026 Luxing Yang.
# Licensed under the MIT License. See LICENSE in the repository root.

"""Root runner for the Note 1 examples."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    """Run a repository command from the root directory."""

    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Note 1 checks, figures, and bounded examples.")
    parser.add_argument(
        "command",
        choices=["smoke", "figures", "train", "mappo", "all"],
        help="Command group to run. Extra arguments are passed to train or mappo.",
    )
    args, rest = parser.parse_known_args()
    py = sys.executable

    if args.command in {"smoke", "all"}:
        run(["bash", "scripts/run_smoke_tests.sh"])
    if args.command in {"figures", "all"}:
        run([py, "scripts/generate_figures.py"])
    if args.command == "train":
        run([py, "scripts/run_training_iterations.py", *rest])
    if args.command == "mappo":
        run([py, "src/node_siprs_mappo.py", *rest])


if __name__ == "__main__":
    main()
