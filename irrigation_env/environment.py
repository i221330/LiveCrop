"""Gymnasium environment: Irrigation-v0.

One growing season (150 days) of daily irrigation decisions over 4 farm
zones. State couples per-zone soil moisture, a short weather forecast,
and the current crop growth stage. Reward trades water cost against
crop stress and waterlogging, with a terminal yield bonus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from irrigation_env.dynamics import (
    KcStages,
    SoilParams,
    crop_kc,
    seasonal_yield,
    soil_step,
)
from irrigation_env.weather import WeatherConfig, generate_season


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "env.yaml"

# Observation normalization maxima (used for scaling into [0, 1]).
_NORM_RAIN_MM = 20.0
_NORM_TMAX_C = 45.0
_NORM_TMIN_C = 30.0
_NORM_ET0_MM = 12.0
_NORM_KC = 1.2


@dataclass(frozen=True)
class EnvConfig:
    num_zones: int
    season_length: int
    start_month: int
    water_levels_mm: np.ndarray
    daily_budget_mm: float
    forecast_horizon: int
    soil: SoilParams
    initial_moisture_theta: float
    zone_moisture_noise_fraction: float  # ±fraction of initial moisture added per zone at reset
    kc_stages: KcStages
    ky_sensitivity: float
    water_cost_per_mm: float
    stress_penalty_weight: float
    waterlogging_penalty_weight: float
    yield_bonus_weight: float

    @property
    def num_actions_per_zone(self) -> int:
        return len(self.water_levels_mm)

    @property
    def total_actions(self) -> int:
        return self.num_actions_per_zone ** self.num_zones


def load_config(path: Optional[Path] = None) -> EnvConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        raw = yaml.safe_load(f)

    zones = raw["zones"]
    soil = SoilParams(
        root_zone_depth_mm=float(zones["root_zone_depth_mm"]),
        field_capacity_theta=float(zones["field_capacity_theta"]),
        wilting_point_theta=float(zones["wilting_point_theta"]),
        saturation_theta=float(zones["saturation_theta"]),
    )
    return EnvConfig(
        num_zones=int(zones["num_zones"]),
        season_length=int(raw["season"]["length_days"]),
        start_month=int(raw["season"]["start_month"]),
        water_levels_mm=np.array(raw["action"]["water_levels_mm"], dtype=np.float32),
        daily_budget_mm=float(raw["action"]["daily_budget_mm"]),
        forecast_horizon=int(raw["forecast"]["horizon_days"]),
        soil=soil,
        initial_moisture_theta=float(zones["initial_moisture_theta"]),
        zone_moisture_noise_fraction=float(zones.get("zone_moisture_noise_fraction", 0.0)),
        kc_stages=[tuple(s) for s in raw["crop"]["kc_stages"]],
        ky_sensitivity=float(raw["crop"]["ky_sensitivity"]),
        water_cost_per_mm=float(raw["reward"]["water_cost_per_mm"]),
        stress_penalty_weight=float(raw["reward"]["stress_penalty_weight"]),
        waterlogging_penalty_weight=float(raw["reward"]["waterlogging_penalty_weight"]),
        yield_bonus_weight=float(raw["reward"]["yield_bonus_weight"]),
    )


def decode_action(action: int, num_zones: int, num_levels: int) -> np.ndarray:
    """Decode a flat Discrete action into per-zone level indices (zone 0 is LSB)."""
    indices = np.empty(num_zones, dtype=np.int64)
    a = int(action)
    for i in range(num_zones):
        indices[i] = a % num_levels
        a //= num_levels
    return indices


class IrrigationEnv(gym.Env):
    """Gymnasium environment for multi-zone irrigation scheduling."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config: Optional[EnvConfig] = None, config_path: Optional[str] = None):
        super().__init__()
        self.cfg = config if config is not None else load_config(config_path)

        obs_dim = (
            self.cfg.num_zones
            + self.cfg.forecast_horizon * 4
            + 2  # Kc, season progress
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.cfg.total_actions)

        self._weather_cfg = WeatherConfig(
            start_month=self.cfg.start_month,
            length_days=self.cfg.season_length,
        )

        # Buffers populated on reset
        self._weather: np.ndarray = np.empty((0, 4), dtype=np.float32)
        self._water: np.ndarray = np.empty(0, dtype=np.float32)
        self._day: int = 0
        self._stress_history: List[np.ndarray] = []
        self._episode_water_used: float = 0.0

    # -- Gym API ------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        # np_random is set by super().reset when seed is provided
        self._weather = generate_season(self.np_random, self._weather_cfg).astype(np.float32)
        initial_w = (
            self.cfg.initial_moisture_theta * self.cfg.soil.root_zone_depth_mm
        )
        if self.cfg.zone_moisture_noise_fraction > 0.0:
            noise = self.np_random.uniform(
                -self.cfg.zone_moisture_noise_fraction,
                self.cfg.zone_moisture_noise_fraction,
                size=self.cfg.num_zones,
            )
            self._water = np.clip(
                initial_w * (1.0 + noise),
                self.cfg.soil.wp_mm,
                self.cfg.soil.sat_mm,
            ).astype(np.float32)
        else:
            self._water = np.full(self.cfg.num_zones, initial_w, dtype=np.float32)
        self._day = 0
        self._stress_history = []
        self._episode_water_used = 0.0
        return self._observation(), self._info_snapshot()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        indices = decode_action(action, self.cfg.num_zones, self.cfg.num_actions_per_zone)
        requested = self.cfg.water_levels_mm[indices]
        total_requested = float(requested.sum())
        if total_requested > self.cfg.daily_budget_mm:
            scale = self.cfg.daily_budget_mm / total_requested
            applied = requested * scale
        else:
            applied = requested.copy()

        rain_mm, tmax, tmin, et0 = self._weather[self._day]
        kc = crop_kc(self._day, self.cfg.kc_stages)
        outcome = soil_step(
            water_mm=self._water,
            rain_mm=float(rain_mm),
            irrig_mm=applied.astype(np.float32),
            et0_mm=float(et0),
            kc=kc,
            params=self.cfg.soil,
        )
        self._water = outcome.water_mm.astype(np.float32)
        self._stress_history.append(outcome.stress.copy())

        water_used = float(applied.sum())
        self._episode_water_used += water_used

        step_reward = (
            -self.cfg.water_cost_per_mm * water_used
            - self.cfg.stress_penalty_weight * float(outcome.stress.mean())
            - self.cfg.waterlogging_penalty_weight * float(outcome.waterlogging.mean())
        )

        self._day += 1
        terminated = self._day >= self.cfg.season_length
        truncated = False

        if terminated:
            history = np.stack(self._stress_history, axis=0)
            yields = seasonal_yield(history, self.cfg.ky_sensitivity)
            terminal_bonus = self.cfg.yield_bonus_weight * float(yields.mean())
            step_reward += terminal_bonus
            info = self._info_snapshot(yields=yields, terminal_bonus=terminal_bonus)
        else:
            info = self._info_snapshot()

        info.update(
            {
                "water_applied_mm": applied,
                "water_requested_mm": requested,
                "step_water_cost": self.cfg.water_cost_per_mm * water_used,
                "step_stress": float(outcome.stress.mean()),
                "step_waterlogging": float(outcome.waterlogging.mean()),
                "kc": kc,
                "rain_mm": float(rain_mm),
                "et0_mm": float(et0),
            }
        )

        return self._observation(), float(step_reward), terminated, truncated, info

    def render(self):  # pragma: no cover - ASCII only
        if self._day == 0:
            print(f"day | {' '.join(f'Z{i}' for i in range(self.cfg.num_zones))} | rain et0")
        w_norm = self._water / self.cfg.soil.sat_mm
        bar = " ".join(f"{v:.2f}" for v in w_norm)
        day_idx = min(self._day, self.cfg.season_length - 1)
        rain, _, _, et0 = self._weather[day_idx]
        print(f"{self._day:3d} | {bar} | {rain:4.1f} {et0:4.2f}")

    # -- Internals ----------------------------------------------------------

    def _observation(self) -> np.ndarray:
        soil_norm = self._water / self.cfg.soil.sat_mm
        obs_parts = [soil_norm]

        for offset in range(self.cfg.forecast_horizon):
            idx = self._day + offset
            if idx < self.cfg.season_length:
                rain, tmax, tmin, et0 = self._weather[idx]
            else:
                rain = tmax = tmin = et0 = 0.0
            obs_parts.append(
                np.array(
                    [
                        min(rain / _NORM_RAIN_MM, 1.0),
                        min(max(tmax, 0.0) / _NORM_TMAX_C, 1.0),
                        min(max(tmin, 0.0) / _NORM_TMIN_C, 1.0),
                        min(et0 / _NORM_ET0_MM, 1.0),
                    ],
                    dtype=np.float32,
                )
            )

        day_idx = min(self._day, self.cfg.season_length - 1)
        kc = crop_kc(day_idx, self.cfg.kc_stages)
        progress = self._day / self.cfg.season_length
        obs_parts.append(np.array([kc / _NORM_KC, progress], dtype=np.float32))

        obs = np.concatenate(obs_parts).astype(np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _info_snapshot(
        self,
        yields: Optional[np.ndarray] = None,
        terminal_bonus: Optional[float] = None,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "day": self._day,
            "water_mm": self._water.copy(),
            "episode_water_used_mm": self._episode_water_used,
        }
        if yields is not None:
            info["zone_yields"] = yields
            info["mean_yield"] = float(yields.mean())
        if terminal_bonus is not None:
            info["terminal_yield_bonus"] = terminal_bonus
        return info
