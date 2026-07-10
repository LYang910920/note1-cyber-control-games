"""
Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

Sampled SIR cyber-defense environment for ODE-RL, DDQN, PPO, SAC, and MADRL examples.

The environment illustrates three control types used in the repository guides:
  * sampled flow control: selected action changes ODE rates over an interval;
  * parameterized/mixed action: a mode plus an intensity held by ZOH;
  * impulse/reset control: selected action causes x(t_k+) = G(x(t_k-),a_k).

The code avoids gym/gymnasium dependencies so that it can be read
as plain Python.  It still follows the familiar reset/step interface.

Timing convention
-----------------
At epoch k, policies observe x(t_k^-), choose actions, optional impulses create
x(t_k^+), the ODE is integrated over [t_k,t_{k+1}), and the next observation is
x(t_{k+1}^-).  RK4 substeps are internal solver points, not extra decisions.
Here t_k denotes learning action/observation points.  If the original model has
its own impulse/event points, denote them by tau_j and decide whether they
coincide with, sit inside, or are chosen separately from the learning grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np
from cybercontrol.models import SampledSIRParams, isolation_jump, sampled_sir_flow_rhs
from cybercontrol.numerics import project_simplex3, rk4_integrate

from .actions import Action, AttackerMode, DefenderMode, ModeIntensityAction


@dataclass
class EnvConfig:
    # dt is the fixed sampled-data decision interval used by the examples.
    # environment: t_k = k * dt.  The tutorial also discusses nonuniform
    # intervals Delta t_k = t_{k+1} - t_k; those require carrying the next
    # action time in the environment state or scheduler.
    dt: float = 1.0
    # substeps are internal RK4 solver steps inside one environment transition.
    # They improve integration accuracy but do not create extra replay items.
    substeps: int = 10
    # horizon is the maximum number of policy decisions in one episode.
    horizon: int = 100
    params: SampledSIRParams = field(default_factory=SampledSIRParams)
    # reward weights
    w_I: float = 10.0
    w_S: float = 0.5
    c_patch: float = 1.0
    c_clean: float = 1.0
    c_deceive: float = 1.5
    # Isolation is an impulsive control.  Its cost is charged once at the jump,
    # not multiplied by the continuous-flow interval dt.
    c_isolate: float = 2.0
    usability_cost: float = 2.5
    attack_costs: Tuple[float, float, float, float] = (0.05, 0.20, 0.30, 0.25)
    randomize_initial_state: bool = False


class SampledContinuousImpulseCyberEnv:
    """Sampled-data SIR propagation with ZOH flow actions and optional impulses.

    Inputs to `step` are a defender action and an attacker action.  The method
    applies any immediate jump, integrates the ODE over one decision interval,
    and returns the next observation, both players' rewards, a terminal flag,
    and diagnostic information for plotting or debugging.  The state shape is
    ``(3,)`` for ``[S,I,R]`` and mass is projected back to the simplex.
    """

    # Defender modes
    DEF_NONE = int(DefenderMode.NONE)
    DEF_PATCH = int(DefenderMode.PATCH)
    DEF_CLEAN = int(DefenderMode.CLEAN)
    DEF_DECEIVE = int(DefenderMode.DECEIVE)
    DEF_ISOLATE = int(DefenderMode.ISOLATE)

    # Attacker modes
    ATK_SCAN = int(AttackerMode.SCAN)
    ATK_EXPLOIT = int(AttackerMode.EXPLOIT)
    ATK_LATERAL = int(AttackerMode.LATERAL)
    ATK_STEALTH = int(AttackerMode.STEALTH)

    def __init__(self, config: EnvConfig | None = None, seed: int = 0):
        self.cfg = config if config is not None else EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.state = np.array([0.95, 0.05, 0.0], dtype=np.float64)

    @property
    def obs_dim(self) -> int:
        return 4

    @property
    def n_defender_actions(self) -> int:
        return 5

    @property
    def n_attacker_actions(self) -> int:
        return 4

    def reset(self, x0=None):
        self.t = 0
        if x0 is not None:
            self.state = project_simplex3(np.array(x0, dtype=np.float64))
        elif self.cfg.randomize_initial_state:
            I0 = self.rng.uniform(0.02, 0.10)
            R0 = self.rng.uniform(0.0, 0.05)
            S0 = max(1e-6, 1.0 - I0 - R0)
            self.state = project_simplex3(np.array([S0, I0, R0], dtype=np.float64))
        else:
            self.state = np.array([0.95, 0.05, 0.0], dtype=np.float64)
        return self.observe()

    def observe(self):
        """Return the observation used by policies.

        For a richer study, append time-to-go, budget remaining, or moving
        averages.  Keeping the default observation equal to x(t_k) makes the
        Markov-game conversion transparent.  The environment uses the pre-jump
        state at the current decision epoch as the observation.
        """
        decision_phase = self.t / max(1, self.cfg.horizon)
        return np.r_[self.state, decision_phase].astype(np.float64)

    def decode_action(self, action: Action) -> Tuple[int, float]:
        """Decode an integer mode or ``ModeIntensityAction``.

        Solver substeps are not actions.  For flow modes, the decoded intensity
        is mapped to rates and held by zero-order hold on ``[t_k,t_{k+1})``.  A
        state reset occurs only when the decoded mode is ``DEF_ISOLATE``.
        """

        if isinstance(action, ModeIntensityAction):
            return int(action.mode), float(np.clip(action.intensity, 0.0, 1.0))
        return int(action), 1.0

    def defense_parameters(self, action: Action) -> Dict[str, float]:
        """Map a sampled defender action to held flow rates and impulse size."""

        mode, v = self.decode_action(action)
        return {
            "patch": 0.30 * v if mode == self.DEF_PATCH else 0.0,
            "clean": 0.25 * v if mode == self.DEF_CLEAN else 0.0,
            "deceive": 0.35 * v if mode == self.DEF_DECEIVE else 0.0,
            "isolate": 0.40 * v if mode == self.DEF_ISOLATE else 0.0,
            "mode": mode,
            "intensity": v,
        }

    def attack_parameters(self, action: Action) -> Dict[str, float]:
        """Map a sampled attacker action to held flow-rate parameters."""

        mode, v = self.decode_action(action)
        attack_boost = [0.10, 0.50, 0.70, 0.20][mode] * v
        stealth_factor = 0.5 if mode == self.ATK_STEALTH else 1.0
        return {
            "beta": self.cfg.params.beta0 * (1.0 + attack_boost),
            "stealth_factor": stealth_factor,
            "mode": mode,
            "intensity": v,
            "cost": self.cfg.attack_costs[mode] * v * v,
        }

    def jump_map(self, x, dpar):
        """Apply immediate jumps before continuous integration.

        This is the implementation difference between an impulsive control and a
        discrete decision that merely changes future ODE rates.  Isolation is
        implemented as a jump because it immediately removes a fraction of the
        compromised population.
        """
        y = x.copy()
        if dpar["mode"] == self.DEF_ISOLATE:
            y = isolation_jump(y, dpar["isolate"])
        return project_simplex3(y)

    def step(self, defender_action: Action, attacker_action: Action = ATK_EXPLOIT):
        """Advance one sampled-data MDP/Markov-game transition.

        The ordering is deliberately explicit:

        1. observe the pre-jump state x(t_k^-);
        2. decode defender and attacker actions;
        3. apply an instantaneous jump, if the chosen mode has one;
        4. integrate the continuous ODE flow over one decision interval;
        5. return x(t_{k+1}^-) as the next observation.

        Reward equation:
        ``r_D = -(dt * running_defense_cost + impulse_cost)`` and
        ``r_A = dt * (8 I_mean + S_mean - attack_cost)``.  Running and impulse
        costs are kept separate in ``info``.
        """
        epoch = self.t
        t_start = epoch * self.cfg.dt
        dpar = self.defense_parameters(defender_action)
        apar = self.attack_parameters(attacker_action)
        pre_jump = self.state.copy()
        post_jump = self.jump_map(pre_jump, dpar)

        def rhs(x, t):
            return sampled_sir_flow_rhs(x, dpar, apar, self.cfg.params)

        next_state, path = rk4_integrate(
            rhs,
            post_jump,
            t0=t_start,
            dt=self.cfg.dt,
            substeps=self.cfg.substeps,
            project=project_simplex3,
        )
        self.state = next_state
        self.t += 1

        I_mean = float(path[:, 1].mean())
        S_mean = float(path[:, 0].mean())
        running_defense_cost = (
            self.cfg.w_I * I_mean
            + self.cfg.w_S * S_mean
            + self.cfg.c_patch * dpar["patch"] ** 2
            + self.cfg.c_clean * dpar["clean"] ** 2
            + self.cfg.c_deceive * dpar["deceive"] ** 2
        )
        removed_by_impulse = max(0.0, float(pre_jump[1] - post_jump[1]))
        impulse_cost = 0.0
        if dpar["mode"] == self.DEF_ISOLATE:
            impulse_cost = (
                self.cfg.c_isolate * dpar["intensity"] ** 2
                + self.cfg.usability_cost * removed_by_impulse
            )
        defender_reward = -(self.cfg.dt * running_defense_cost + impulse_cost)
        attacker_reward = float(self.cfg.dt * (8.0 * I_mean + 1.0 * S_mean - apar["cost"]))
        done = bool(self.t >= self.cfg.horizon or next_state[1] < 1e-5 or next_state[1] > 0.95)
        info = {
            "decision_epoch": epoch,
            "t_observe": t_start,
            "t_pre_jump": t_start,
            "t_post_jump": t_start,
            "t_next_observe": t_start + self.cfg.dt,
            "decision_dt": self.cfg.dt,
            "solver_substeps": self.cfg.substeps,
            "transition_order": "observe -> jump_map -> ODE flow -> next_observation",
            "observation_state": "pre_jump_state_at_t_k_minus",
            "next_observation_state": "state_at_t_k_plus_1_minus",
            "jump_applied": bool(np.linalg.norm(post_jump - pre_jump, ord=1) > 1e-12),
            "running_defense_cost": running_defense_cost,
            "impulse_cost": impulse_cost,
            "removed_by_impulse": removed_by_impulse,
            "pre_jump": pre_jump,
            "post_jump": post_jump,
            "path": path,
            "dpar": dpar,
            "apar": apar,
        }
        return (
            self.observe(),
            {"defender": defender_reward, "attacker": attacker_reward},
            done,
            info,
        )


def scripted_attacker(env: SampledContinuousImpulseCyberEnv, k: int):
    """A non-learning attacker used for first defender experiments."""
    if k < 20:
        return env.ATK_SCAN
    if k < 60:
        return env.ATK_EXPLOIT
    return env.ATK_LATERAL
