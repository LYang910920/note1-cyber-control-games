#!/usr/bin/env bash
# Copyright (c) 2026 Luxing Yang.
# Licensed under the MIT License. See LICENSE in the repository root.

set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"

"${PYTHON_BIN}" src/cyber_dynamics.py
"${PYTHON_BIN}" src/cyber_hybrid_env.py
"${PYTHON_BIN}" src/scenario_profiles.py
"${PYTHON_BIN}" src/fbsm_malware_baseline.py --smoke
"${PYTHON_BIN}" src/ddqn_cyber_defense.py --smoke
"${PYTHON_BIN}" src/madrl_ctde_hybrid_game.py --smoke
"${PYTHON_BIN}" src/node_siprs_mappo.py --smoke --device cpu
"${PYTHON_BIN}" -m unittest discover -s tests
