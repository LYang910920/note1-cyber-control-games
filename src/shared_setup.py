"""Import checks for the shared foundation package used by Note 1."""

from __future__ import annotations


def ensure_foundation_package() -> None:
    """Fail with a clear setup message when ``cybercontrol`` is unavailable."""

    try:
        import cybercontrol  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Note 1 reuses the shared foundation package. From this repository, run:\n"
            "  python -m pip install -e ../network-control-differential-games[torch,dev]\n"
            "  python -m pip install -e .[dev]\n"
            "Then rerun the example."
        ) from exc
