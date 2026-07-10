#!/usr/bin/env bash
# Copyright (c) 2026 Luxing Yang.
# Licensed under the MIT License. See LICENSE in the repository root.

set -euo pipefail

PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -x ".venv/bin/python" ]]; then
        PYTHON_BIN=".venv/bin/python"
    elif [[ -x "../.venv/bin/python" ]]; then
        PYTHON_BIN="../.venv/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

"${PYTHON_BIN}" -m cybergames smoke
"${PYTHON_BIN}" -m pytest -q
