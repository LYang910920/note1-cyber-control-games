"""Helpers for importing the shared foundation package during local teaching runs."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_foundation_package() -> None:
    """Add the sibling foundation package to ``sys.path`` when not installed.

    In the recommended workspace layout, students can run examples before or
    after ``pip install -e ../network-control-differential-games``.  This helper
    keeps both paths working while the README still documents the editable
    install as the clean option.
    """

    foundation_src = Path(__file__).resolve().parents[2] / "network-control-differential-games" / "src"
    if foundation_src.exists() and str(foundation_src) not in sys.path:
        sys.path.insert(0, str(foundation_src))
