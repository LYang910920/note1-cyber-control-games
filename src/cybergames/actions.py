"""Typed sampled-flow, parameterized, and impulse action definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Union


class DefenderMode(IntEnum):
    """Defender mode selected at a sampled decision epoch."""

    NONE = 0
    PATCH = 1
    CLEAN = 2
    DECEIVE = 3
    ISOLATE = 4


class AttackerMode(IntEnum):
    """Attacker flow mode held over one decision interval."""

    SCAN = 0
    EXPLOIT = 1
    LATERAL = 2
    STEALTH = 3


@dataclass(frozen=True)
class ModeIntensityAction:
    """Parameterized action ``(mode, intensity)`` with intensity in ``[0,1]``."""

    mode: int | DefenderMode | AttackerMode
    intensity: float = 1.0


Action = Union[int, DefenderMode, AttackerMode, ModeIntensityAction]


def mode_intensity(
    mode: int | DefenderMode | AttackerMode, intensity: float = 1.0
) -> ModeIntensityAction:
    """Build a parameterized action without implying an impulse reset."""

    return ModeIntensityAction(mode=mode, intensity=float(intensity))
