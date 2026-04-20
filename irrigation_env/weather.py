"""Synthetic daily weather for a Fresno, CA growing season.

Monthly climate means are calibrated to published NOAA/NASA POWER
statistics for Fresno (lat 36.75 N). Daily variation is generated with
an AR(1) process for temperatures and a Bernoulli-Exponential process
for precipitation. Reference evapotranspiration (ET0) is computed from
the Hargreaves-Samani equation, which depends only on temperature and
extraterrestrial solar radiation, keeping the generator self-contained.

Returns a per-season array with columns: precipitation [mm], tmax [C],
tmin [C], ET0 [mm/day].
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Tuple

import numpy as np


LATITUDE_DEG = 36.75  # Fresno, CA

# Monthly midpoint climatology for the growing season. Values are derived
# from 1991-2020 normals and NASA POWER daily aggregates.
# (doy, tmax_mean_C, tmin_mean_C, rainy_day_prob, mean_precip_per_rainy_day_mm)
_CLIMATE_POINTS: Tuple[Tuple[int, float, float, float, float], ...] = (
    (121, 26.0, 11.0, 0.10, 4.5),   # May 1
    (135, 28.0, 12.0, 0.08, 4.0),   # May 15
    (166, 33.0, 16.0, 0.03, 3.0),   # Jun 15
    (196, 36.0, 19.0, 0.01, 3.0),   # Jul 15
    (227, 35.0, 18.0, 0.01, 3.0),   # Aug 15
    (258, 32.0, 15.0, 0.05, 3.0),   # Sep 15
    (270, 30.0, 13.0, 0.07, 3.5),   # Sep 27
)


@dataclass(frozen=True)
class WeatherConfig:
    start_month: int = 5
    start_day: int = 1
    length_days: int = 150
    latitude_deg: float = LATITUDE_DEG
    tmax_ar_phi: float = 0.7
    tmin_ar_phi: float = 0.7
    tmax_sigma: float = 3.0
    tmin_sigma: float = 2.0
    min_diurnal_range: float = 4.0  # enforce Tmax >= Tmin + this


def _interp_climatology(doys: np.ndarray) -> np.ndarray:
    """Interpolate (tmax_mean, tmin_mean, p_rain, precip_mean) for given DOYs."""
    pts = np.array(_CLIMATE_POINTS)
    ref_doy = pts[:, 0]
    out = np.empty((len(doys), 4))
    for i, col in enumerate((1, 2, 3, 4)):
        out[:, i] = np.interp(doys, ref_doy, pts[:, col])
    return out


def extraterrestrial_radiation_mm(doys: np.ndarray, latitude_deg: float) -> np.ndarray:
    """Compute Ra (FAO-56 eq. 21) in mm/day equivalent for given DOYs."""
    lat = np.deg2rad(latitude_deg)
    dr = 1.0 + 0.033 * np.cos(2 * np.pi / 365 * doys)
    decl = 0.409 * np.sin(2 * np.pi / 365 * doys - 1.39)
    ws_arg = np.clip(-np.tan(lat) * np.tan(decl), -1.0, 1.0)
    omega_s = np.arccos(ws_arg)
    gsc = 0.0820  # MJ m^-2 min^-1
    ra_mj = (24 * 60 / np.pi) * gsc * dr * (
        omega_s * np.sin(lat) * np.sin(decl)
        + np.cos(lat) * np.cos(decl) * np.sin(omega_s)
    )
    return ra_mj / 2.45  # MJ/m^2/day -> mm/day equivalent


def _hargreaves_et0(tmax: np.ndarray, tmin: np.ndarray, ra_mm: np.ndarray) -> np.ndarray:
    tmean = 0.5 * (tmax + tmin)
    diurnal = np.maximum(tmax - tmin, 0.0)
    return 0.0023 * (tmean + 17.8) * np.sqrt(diurnal) * ra_mm


def season_doys(cfg: WeatherConfig, year: int = 2023) -> np.ndarray:
    """Day-of-year array for a season starting at (year, start_month, start_day)."""
    start = date(year, cfg.start_month, cfg.start_day)
    return np.array(
        [(start + timedelta(days=i)).timetuple().tm_yday for i in range(cfg.length_days)]
    )


def generate_season(
    rng: np.random.Generator, cfg: WeatherConfig = WeatherConfig()
) -> np.ndarray:
    """Generate one growing-season weather array.

    Returns: np.ndarray of shape (length_days, 4) with columns
    [precip_mm, tmax_C, tmin_C, et0_mm].
    """
    doys = season_doys(cfg)
    clim = _interp_climatology(doys)  # (N, 4): tmax_mean, tmin_mean, p_rain, precip_mean
    tmax_mean, tmin_mean, p_rain, precip_mean = clim.T

    n = len(doys)
    tmax = np.empty(n)
    tmin = np.empty(n)
    tmax_prev = 0.0
    tmin_prev = 0.0
    for t in range(n):
        eps_mx = rng.normal(0.0, cfg.tmax_sigma)
        eps_mn = rng.normal(0.0, cfg.tmin_sigma)
        if t == 0:
            tmax[t] = tmax_mean[t] + eps_mx
            tmin[t] = tmin_mean[t] + eps_mn
        else:
            tmax[t] = tmax_mean[t] + cfg.tmax_ar_phi * tmax_prev + eps_mx
            tmin[t] = tmin_mean[t] + cfg.tmin_ar_phi * tmin_prev + eps_mn
        tmax_prev = tmax[t] - tmax_mean[t]
        tmin_prev = tmin[t] - tmin_mean[t]

    # Enforce physical diurnal ordering
    tmin = np.minimum(tmin, tmax - cfg.min_diurnal_range)

    # Precipitation: Bernoulli-Exponential
    rainy = rng.random(n) < p_rain
    amount = rng.exponential(precip_mean)
    precip = np.where(rainy, amount, 0.0)

    ra = extraterrestrial_radiation_mm(doys, cfg.latitude_deg)
    et0 = _hargreaves_et0(tmax, tmin, ra)
    et0 = np.clip(et0, 0.5, 12.0)

    return np.column_stack([precip, tmax, tmin, et0])


def generate_season_seeded(seed: int, cfg: WeatherConfig = WeatherConfig()) -> np.ndarray:
    return generate_season(np.random.default_rng(seed), cfg)


if __name__ == "__main__":
    import argparse
    import csv
    import sys

    parser = argparse.ArgumentParser(description="Dump a synthetic season weather CSV.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="-", help="Output path, or - for stdout")
    args = parser.parse_args()

    data = generate_season_seeded(args.seed)
    out = sys.stdout if args.out == "-" else open(args.out, "w", newline="")
    writer = csv.writer(out)
    writer.writerow(["day", "precip_mm", "tmax_c", "tmin_c", "et0_mm"])
    for i, row in enumerate(data):
        writer.writerow([i, f"{row[0]:.2f}", f"{row[1]:.2f}", f"{row[2]:.2f}", f"{row[3]:.2f}"])
    if out is not sys.stdout:
        out.close()
