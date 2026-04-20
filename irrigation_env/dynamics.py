"""Soil water balance and crop yield dynamics.

Implements a simplified per-zone soil water balance inspired by FAO-56
irrigation and drainage paper 56, coupled with an FAO-33 style yield
response to water stress. All water quantities are stored in mm of
equivalent depth over the root zone, so they can be added and subtracted
directly regardless of zone area.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class SoilParams:
    root_zone_depth_mm: float = 600.0
    field_capacity_theta: float = 0.35
    wilting_point_theta: float = 0.10
    saturation_theta: float = 0.45
    drainage_tau_days: float = 2.0  # exponential drainage above field capacity

    @property
    def fc_mm(self) -> float:
        return self.field_capacity_theta * self.root_zone_depth_mm

    @property
    def wp_mm(self) -> float:
        return self.wilting_point_theta * self.root_zone_depth_mm

    @property
    def sat_mm(self) -> float:
        return self.saturation_theta * self.root_zone_depth_mm

    @property
    def taw_mm(self) -> float:
        return self.fc_mm - self.wp_mm


@dataclass(frozen=True)
class StepOutcome:
    """Per-zone outcomes from one daily soil-water balance step."""

    water_mm: np.ndarray       # soil water after the step (mm)
    eta_mm: np.ndarray         # actual crop ET (mm)
    runoff_mm: np.ndarray      # water lost to runoff (over saturation)
    drainage_mm: np.ndarray    # water lost to drainage below root zone
    stress: np.ndarray         # 1 - Ks, per zone, in [0, 1]
    waterlogging: np.ndarray   # (W - FC) / (SAT - FC) clipped to [0, 1]


def soil_step(
    water_mm: np.ndarray,
    rain_mm: float,
    irrig_mm: np.ndarray,
    et0_mm: float,
    kc: float,
    params: SoilParams,
) -> StepOutcome:
    """Advance one daily soil water balance step across all zones.

    water_mm and irrig_mm are per-zone arrays; rain and ET0 are scalar
    daily values applied identically to every zone.
    """
    fc, wp, sat, taw = params.fc_mm, params.wp_mm, params.sat_mm, params.taw_mm

    inflow = rain_mm + irrig_mm
    w_pre = water_mm + inflow
    runoff = np.maximum(w_pre - sat, 0.0)
    w_pre = np.minimum(w_pre, sat)

    etc = kc * et0_mm  # potential crop ET (mm/day)
    p = np.clip(0.5 + 0.04 * (5.0 - etc), 0.1, 0.8)
    raw = p * taw

    depletion = np.maximum(fc - w_pre, 0.0)
    # Ks follows the piecewise-linear FAO-56 form
    ks = np.where(
        depletion <= raw,
        1.0,
        np.clip((taw - depletion) / np.maximum(taw - raw, 1e-6), 0.0, 1.0),
    )

    eta = ks * etc
    w_after_et = np.maximum(w_pre - eta, 0.0)

    excess = np.maximum(w_after_et - fc, 0.0)
    drainage = excess * (1.0 - np.exp(-1.0 / params.drainage_tau_days))
    w_new = w_after_et - drainage

    waterlogging = np.clip((w_new - fc) / max(sat - fc, 1e-6), 0.0, 1.0)
    stress = 1.0 - ks

    return StepOutcome(
        water_mm=w_new,
        eta_mm=eta,
        runoff_mm=runoff,
        drainage_mm=drainage,
        stress=stress,
        waterlogging=waterlogging,
    )


KcStages = List[Tuple[int, float]]


def crop_kc(day_in_season: int, stages: KcStages) -> float:
    """Piecewise-linear crop coefficient Kc over the season."""
    days = np.array([s[0] for s in stages], dtype=float)
    kcs = np.array([s[1] for s in stages], dtype=float)
    return float(np.interp(day_in_season, days, kcs))


def seasonal_yield(stress_history: np.ndarray, ky: float) -> np.ndarray:
    """Aggregate per-zone relative yield from a daily stress trajectory.

    stress_history: (days, zones), each element = 1 - Ks in [0, 1].
    Returns: per-zone relative yield in [0, 1] using Y_rel = 1 - Ky * mean(stress).
    """
    mean_stress = stress_history.mean(axis=0)
    return np.clip(1.0 - ky * mean_stress, 0.0, 1.0)
