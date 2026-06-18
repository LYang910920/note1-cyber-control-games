#!/usr/bin/env bash
# Copyright (c) 2026 Luxing Yang.
# Licensed under the MIT License. See LICENSE in the repository root.

set -euo pipefail

python src/cyber_dynamics.py
python src/cyber_hybrid_env.py
python src/scenario_profiles.py
python src/fbsm_malware_baseline.py --smoke
python src/ddqn_cyber_defense.py --smoke
python src/madrl_ctde_hybrid_game.py --smoke
python -m unittest discover -s tests
