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


def resolve_torch_device(configure_torch_fn, *, seed=None, device="auto", threads=1):
    """Resolve ``auto`` consistently across released and in-flight foundation versions.

    The Note repositories may be tested against the foundation package on
    GitHub `main`, while local development often uses the sibling checkout.
    This wrapper keeps the note scripts independent of that ordering.
    """

    requested = "auto" if device is None else str(device)
    torch, _, dtype = configure_torch_fn(seed=seed, device="cpu", threads=threads)

    if requested == "auto":
        if torch.cuda.is_available():
            resolved = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            resolved = "mps"
        else:
            resolved = "cpu"
    else:
        resolved = requested
        if resolved == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        if resolved == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is None or not torch.backends.mps.is_available():
                raise RuntimeError("Apple MPS was requested, but torch.backends.mps.is_available() is false.")
        if resolved not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported torch device {resolved!r}; use 'auto', 'cpu', 'cuda', or 'mps'.")

    return torch, resolved, dtype
