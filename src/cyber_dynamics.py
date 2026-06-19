"""
Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

Compatibility wrapper for shared cyber-dynamics utilities.

The canonical implementations now live in the foundation package
`cybercontrol`.  This module keeps the original Note 1 import path readable for
students and older notebooks:

    from cyber_dynamics import MalwareParams, rk4_integrate
"""
from __future__ import annotations

import numpy as np

from shared_setup import ensure_foundation_package

ensure_foundation_package()

from cybercontrol.models import HybridParams, MalwareParams, controlled_sir_rhs, hybrid_rhs, isolation_jump
from cybercontrol.numerics import project_simplex3, rk4_integrate, rk4_step

Array = np.ndarray

__all__ = [
    "Array",
    "HybridParams",
    "MalwareParams",
    "controlled_sir_rhs",
    "hybrid_rhs",
    "isolation_jump",
    "project_simplex3",
    "rk4_integrate",
    "rk4_step",
]


if __name__ == "__main__":
    # Small sanity check.
    p = MalwareParams()
    x0 = np.array([0.95, 0.05, 0.0])
    rhs = lambda x, t: controlled_sir_rhs(x, u_patch=0.1, u_clean=0.2, p=p)
    xT, path = rk4_integrate(rhs, x0, t0=0.0, dt=1.0, substeps=20, project=project_simplex3)
    print("xT=", xT, "mass=", xT.sum())
