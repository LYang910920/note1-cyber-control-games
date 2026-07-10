"""Deprecated root entry point; use the cybergames package CLI."""

from __future__ import annotations

import sys
import warnings

from cybergames.cli import main as cli_main


def main() -> None:
    """Forward legacy invocations to the public package CLI."""

    warnings.warn(
        "run_all.py is deprecated; use 'python -m cybergames <command>'",
        DeprecationWarning,
        stacklevel=2,
    )
    cli_main(sys.argv[1:] or ["all"])


if __name__ == "__main__":
    main()
