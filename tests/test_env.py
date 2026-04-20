"""Correctness and API-compliance tests for Irrigation-v0."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import irrigation_env  # noqa: F401 - registers Irrigation-v0
from irrigation_env.dynamics import SoilParams, crop_kc, seasonal_yield, soil_step
from irrigation_env.environment import IrrigationEnv, decode_action, load_config
from irrigation_env.weather import generate_season_seeded


# --- API compliance ---------------------------------------------------------


def test_gymnasium_api_compliance():
    """Environment must satisfy the standard Gymnasium contract."""
    env = gym.make("Irrigation-v0", disable_env_checker=True).unwrapped
    check_env(env, skip_render_check=True)


def test_spaces_and_obs_bounds():
    env = IrrigationEnv()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert env.action_space.n == 256
    # Obs must stay normalized across random episodes
    for _ in range(5):
        obs, _ = env.reset(seed=None)
        for _ in range(env.cfg.season_length):
            a = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(a)
            assert env.observation_space.contains(obs), obs
            if term or trunc:
                break


def test_episode_length_matches_config():
    env = IrrigationEnv()
    env.reset(seed=0)
    for t in range(env.cfg.season_length):
        _, _, term, trunc, _ = env.step(0)
        if term or trunc:
            assert t + 1 == env.cfg.season_length
            return
    pytest.fail("Episode did not terminate in expected number of steps")


# --- Determinism ------------------------------------------------------------


def _run_episode(seed: int, actions: np.ndarray) -> np.ndarray:
    env = IrrigationEnv()
    env.reset(seed=seed)
    rewards = []
    for a in actions:
        _, r, term, trunc, _ = env.step(int(a))
        rewards.append(r)
        if term or trunc:
            break
    return np.array(rewards)


def test_seed_determinism():
    rng = np.random.default_rng(42)
    actions = rng.integers(0, 256, size=150)
    r1 = _run_episode(seed=7, actions=actions)
    r2 = _run_episode(seed=7, actions=actions)
    np.testing.assert_allclose(r1, r2)


def test_different_seeds_diverge():
    rng = np.random.default_rng(42)
    actions = rng.integers(0, 256, size=150)
    r1 = _run_episode(seed=7, actions=actions)
    r2 = _run_episode(seed=8, actions=actions)
    assert not np.allclose(r1, r2)


# --- Invariants on step output ---------------------------------------------


def test_budget_cap_enforced():
    env = IrrigationEnv()
    env.reset(seed=0)
    # action=255 means every zone at max level -> total request > budget
    _, _, _, _, info = env.step(255)
    total_applied = float(info["water_applied_mm"].sum())
    assert total_applied <= env.cfg.daily_budget_mm + 1e-6
    assert float(info["water_requested_mm"].sum()) > env.cfg.daily_budget_mm


def test_water_stays_in_physical_range():
    env = IrrigationEnv()
    env.reset(seed=0)
    sat = env.cfg.soil.sat_mm
    for _ in range(150):
        a = env.action_space.sample()
        _, _, term, _, info = env.step(a)
        w = info["water_mm"]
        assert (w >= -1e-6).all() and (w <= sat + 1e-6).all()
        if term:
            break


def test_always_off_yield_below_always_max():
    """Under-irrigation must hurt yield more than over-irrigation."""
    env_off = IrrigationEnv()
    env_off.reset(seed=0)
    for _ in range(env_off.cfg.season_length):
        _, _, term, _, info_off = env_off.step(0)
        if term:
            break

    env_max = IrrigationEnv()
    env_max.reset(seed=0)
    for _ in range(env_max.cfg.season_length):
        _, _, term, _, info_max = env_max.step(env_max.action_space.n - 1)
        if term:
            break

    assert info_off["mean_yield"] < info_max["mean_yield"]


# --- Action decoding --------------------------------------------------------


def test_decode_action_roundtrip():
    # zone 0 is LSB -> a = sum(i_z * 4^z)
    assert list(decode_action(0, 4, 4)) == [0, 0, 0, 0]
    assert list(decode_action(1, 4, 4)) == [1, 0, 0, 0]
    assert list(decode_action(4, 4, 4)) == [0, 1, 0, 0]
    assert list(decode_action(255, 4, 4)) == [3, 3, 3, 3]


# --- Dynamics unit tests ----------------------------------------------------


def test_soil_step_decays_without_inflow():
    params = SoilParams()
    w = np.array([params.fc_mm] * 4)
    out = soil_step(w, rain_mm=0.0, irrig_mm=np.zeros(4), et0_mm=6.0, kc=1.0, params=params)
    assert (out.water_mm < params.fc_mm).all()
    assert (out.stress >= 0.0).all() and (out.stress <= 1.0).all()


def test_soil_step_saturation_causes_runoff():
    params = SoilParams()
    w = np.full(4, params.sat_mm)
    out = soil_step(w, rain_mm=50.0, irrig_mm=np.zeros(4), et0_mm=5.0, kc=1.0, params=params)
    assert (out.runoff_mm > 0).all()
    assert (out.water_mm <= params.sat_mm + 1e-6).all()


def test_yield_aggregation_bounds():
    stress = np.zeros((150, 4))
    y = seasonal_yield(stress, ky=0.4)
    np.testing.assert_allclose(y, 1.0)
    stress = np.ones((150, 4))
    y = seasonal_yield(stress, ky=0.4)
    np.testing.assert_allclose(y, 0.6)


def test_crop_kc_piecewise_linear():
    stages = [(0, 0.4), (20, 0.4), (50, 1.15), (100, 1.15), (150, 0.6)]
    assert crop_kc(0, stages) == pytest.approx(0.4)
    assert crop_kc(35, stages) == pytest.approx(0.4 + (1.15 - 0.4) * (35 - 20) / 30)
    assert crop_kc(75, stages) == pytest.approx(1.15)
    assert crop_kc(150, stages) == pytest.approx(0.6)


# --- Weather ----------------------------------------------------------------


def test_weather_invariants():
    data = generate_season_seeded(0)
    assert data.shape == (150, 4)
    assert (data[:, 0] >= 0).all()                # precip non-negative
    assert (data[:, 1] >= data[:, 2]).all()       # tmax >= tmin
    assert (data[:, 3] > 0).all()                 # ET0 positive
    # Different seeds diverge
    assert not np.allclose(data, generate_season_seeded(1))


# --- Config -----------------------------------------------------------------


def test_load_config_produces_valid_spec():
    cfg = load_config()
    assert cfg.num_zones == 4
    assert cfg.total_actions == 256
    assert cfg.soil.fc_mm > cfg.soil.wp_mm
    assert cfg.soil.sat_mm > cfg.soil.fc_mm
