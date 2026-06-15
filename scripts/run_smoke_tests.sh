#!/usr/bin/env bash
set -euo pipefail

python src/cyber_dynamics.py
python src/cyber_hybrid_env.py
python src/fbsm_malware_baseline.py --smoke
python src/ddqn_cyber_defense.py --smoke
python src/madrl_ctde_hybrid_game.py --smoke
python -m unittest discover -s tests
