"""
Shared cyber-dynamics utilities for the lecture-note examples.

The goal is clarity rather than maximum speed.  The functions implement:
  1. projection of compartment states onto the probability simplex;
  2. RK4 integration for continuous-time ODE dynamics;
  3. a controlled malware SIR model;
  4. a hybrid malware/deception model with continuous flow and optional jumps.

All states are proportions.  For SIR-style models, the first three components
should sum to one.  The helper functions renormalize small numerical drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import numpy as np

Array = np.ndarray


def project_simplex3(x: Array) -> Array:
    """Project the first three entries of x to a nonnegative unit simplex.

    This is not a mathematically exact Euclidean projection.  It is a practical
    renormalization used after ODE steps to remove tiny negative values and drift.
    """
    y = np.array(x, dtype=np.float64).copy()
    y[:3] = np.maximum(y[:3], 0.0)
    total = float(y[:3].sum())
    if total <= 1e-12:
        y[:3] = np.array([1.0, 0.0, 0.0])
    else:
        y[:3] /= total
    if len(y) > 3:
        y[3:] = np.clip(y[3:], 0.0, 1.0)
    return y


def rk4_integrate(
    rhs: Callable[[Array, float], Array],
    x0: Array,
    t0: float,
    dt: float,
    substeps: int = 10,
    project: Callable[[Array], Array] | None = None,
) -> Tuple[Array, Array]:
    """Integrate x' = rhs(x,t) from t0 to t0+dt by RK4.

    Parameters
    ----------
    rhs:
        Function taking state and time and returning dx/dt.
    x0:
        Initial state at t0.
    dt:
        Length of integration interval.
    substeps:
        Number of smaller RK4 steps.  Increase this if dynamics are stiff or if
        state mass drifts noticeably.
    project:
        Optional projection/renormalization applied after every substep.

    Returns
    -------
    xT:
        Final state.
    path:
        Array containing states at each substep, including x0.
    """
    h = dt / float(substeps)
    y = np.array(x0, dtype=np.float64).copy()
    path = [y.copy()]
    t = t0
    for _ in range(substeps):
        k1 = rhs(y, t)
        k2 = rhs(y + 0.5 * h * k1, t + 0.5 * h)
        k3 = rhs(y + 0.5 * h * k2, t + 0.5 * h)
        k4 = rhs(y + h * k3, t + h)
        y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        if project is not None:
            y = project(y)
        t += h
        path.append(y.copy())
    return y, np.asarray(path)


@dataclass
class MalwareParams:
    beta: float = 0.65      # compromise rate
    gamma: float = 0.05     # natural recovery/removal
    omega: float = 0.01     # loss of immunity/protection


def controlled_sir_rhs(x: Array, u_patch: float, u_clean: float, p: MalwareParams) -> Array:
    """Continuous-control malware propagation model.

    State x=[S,I,R].  u_patch moves vulnerable devices S to protected/recovered
    R.  u_clean cleans compromised devices I and moves them to R.
    """
    S, I, R = x[:3]
    dS = -p.beta * S * I - u_patch * S + p.omega * R
    dI = p.beta * S * I - (p.gamma + u_clean) * I
    dR = u_patch * S + (p.gamma + u_clean) * I - p.omega * R
    return np.array([dS, dI, dR], dtype=np.float64)


@dataclass
class HybridParams:
    beta0: float = 0.65
    gamma: float = 0.05
    omega: float = 0.01
    chi: float = 0.70      # deception effectiveness
    xi: float = 0.04       # natural deception decay
    zeta: float = 0.08     # attacker learning against deception


def hybrid_rhs(x: Array, dpar: Dict[str, float], apar: Dict[str, float], p: HybridParams) -> Array:
    """Hybrid malware/deception flow between decision epochs.

    State x=[S,I,R,z].  The discrete action has already been decoded into
    continuous rates in dpar and apar.  The flow is continuous over an interval.
    Any instantaneous jump should be applied before calling this RHS.
    """
    S, I, R, z = x
    beta = apar.get("beta", p.beta0)
    clean = dpar.get("clean", 0.0) * apar.get("stealth_factor", 1.0)
    patch = dpar.get("patch", 0.0)
    deceive = dpar.get("deceive", 0.0)
    effective_beta = beta * max(0.0, 1.0 - p.chi * z)
    dS = -effective_beta * S * I - patch * S + p.omega * R
    dI = effective_beta * S * I - (p.gamma + clean) * I
    dR = patch * S + (p.gamma + clean) * I - p.omega * R
    dz = deceive * (1.0 - z) - p.xi * z - apar.get("deception_learning", 0.0) * z
    return np.array([dS, dI, dR, dz], dtype=np.float64)


if __name__ == "__main__":
    # Small sanity check.
    p = MalwareParams()
    x0 = np.array([0.95, 0.05, 0.0])
    rhs = lambda x, t: controlled_sir_rhs(x, u_patch=0.1, u_clean=0.2, p=p)
    xT, path = rk4_integrate(rhs, x0, t0=0.0, dt=1.0, substeps=20, project=project_simplex3)
    print("xT=", xT, "mass=", xT.sum())
