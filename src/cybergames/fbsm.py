"""
Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

Forward-backward sweep method (FBSM) for a controlled malware model.

Problem
-------
Minimize
    J = int_0^T [ A I(t) + (B/2) u(t)^2 ] dt + A_T I(T)
subject to
    S' = -beta S I - u S
    I' =  beta S I - gamma I
    R' =  gamma I + u S
    0 <= u(t) <= u_max.

This is an open-loop continuous-control baseline.  It is useful for comparison
with RL because it solves the PMP optimality system for a fixed initial state.
"""

from __future__ import annotations

import logging
import numpy as np

from cybercontrol.numerics import project_simplex3, trapezoid

LOGGER = logging.getLogger(__name__)


def state_rhs(x, u, beta, gamma):
    S, I, R = x
    return np.array(
        [-beta * S * I - u * S, beta * S * I - gamma * I, gamma * I + u * S], dtype=np.float64
    )


def costate_rhs(x, lam, u, beta, gamma, A):
    """Return lambda' = -partial H/partial x.

    H = A I + 0.5 B u^2 + lambda_S(-beta S I - u S)
        + lambda_I(beta S I - gamma I) + lambda_R(gamma I + u S)
    """
    S, I, R = x
    lS, lI, lR = lam
    dH_dS = lS * (-beta * I - u) + lI * (beta * I) + lR * u
    dH_dI = A + lS * (-beta * S) + lI * (beta * S - gamma) + lR * gamma
    dH_dR = 0.0
    return -np.array([dH_dS, dH_dI, dH_dR], dtype=np.float64)


def rk4_state_step(x, u, h, beta, gamma):
    def f(y):
        return state_rhs(y, u, beta, gamma)

    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 = f(x + h * k3)
    y = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return project_simplex3(y)


def rk4_costate_step_backward(x, lam, u, h, beta, gamma, A):
    """One backward RK4 step from t_k to t_{k-1} with step length h."""

    def f(ell):
        return costate_rhs(x, ell, u, beta, gamma, A)

    # integrate lambda' = f(lambda) backward by using -h
    hh = -h
    k1 = f(lam)
    k2 = f(lam + 0.5 * hh * k1)
    k3 = f(lam + 0.5 * hh * k2)
    k4 = f(lam + hh * k3)
    return lam + hh * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def optimal_control_from_stationarity(x, lam, B, u_max):
    """Stationarity: dH/du = B u + S(-lambda_S+lambda_R)=0."""
    S = x[0]
    lS, _, lR = lam
    u_unc = S * (lS - lR) / B
    return float(np.clip(u_unc, 0.0, u_max))


def solve_fbsm(
    T=40.0,
    n=400,
    beta=0.8,
    gamma=0.15,
    A=10.0,
    B=1.0,
    A_terminal=20.0,
    u_max=1.0,
    max_iter=200,
    relax=0.5,
    tol=1e-5,
    return_history=False,
):
    """Solve the PMP state/costate/control loop by forward-backward sweeps.

    The output is the time grid, state trajectory, open-loop control, costate
    trajectory, objective value, and optionally per-iteration diagnostics.  Use
    the diagnostics to check whether the control update has stabilized.
    """
    t = np.linspace(0.0, T, n + 1)
    h = T / n
    x = np.zeros((n + 1, 3), dtype=np.float64)
    lam = np.zeros((n + 1, 3), dtype=np.float64)
    u = np.zeros(n + 1, dtype=np.float64)
    x0 = np.array([0.95, 0.05, 0.0], dtype=np.float64)
    history = []

    for it in range(max_iter):
        x[0] = x0
        for k in range(n):
            x[k + 1] = rk4_state_step(x[k], u[k], h, beta, gamma)

        # terminal condition from terminal penalty A_T I(T)
        lam[n] = np.array([0.0, A_terminal, 0.0])
        for k in range(n, 0, -1):
            lam[k - 1] = rk4_costate_step_backward(x[k], lam[k], u[k], h, beta, gamma, A)

        u_new = np.zeros_like(u)
        for k in range(n + 1):
            u_new[k] = optimal_control_from_stationarity(x[k], lam[k], B, u_max)
        diff = np.max(np.abs(u_new - u))
        u = relax * u_new + (1.0 - relax) * u
        running_cost = A * x[:, 1] + 0.5 * B * u * u
        objective = trapezoid(running_cost, t) + A_terminal * x[-1, 1]
        history.append(
            {
                "iteration": it,
                "max_control_change": float(diff),
                "peak_compromised": float(x[:, 1].max()),
                "mean_control": float(u.mean()),
                "objective": float(objective),
            }
        )
        if it % 10 == 0:
            LOGGER.info(
                "iter=%03d max_control_change=%.4e peak_I=%.4f",
                it,
                diff,
                x[:, 1].max(),
            )
        if diff < tol:
            LOGGER.info("converged at iteration %d", it)
            break

    running_cost = A * x[:, 1] + 0.5 * B * u * u
    J = trapezoid(running_cost, t) + A_terminal * x[-1, 1]
    if return_history:
        return t, x, u, lam, J, history
    return t, x, u, lam, J
