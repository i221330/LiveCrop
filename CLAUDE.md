# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A custom Gymnasium environment (`Irrigation-v0`) simulating multi-zone irrigation scheduling, plus a planned benchmark of DQN, PPO, and A2C from Stable-Baselines3. The portfolio artifact is the environment itself — the algorithms are off-the-shelf.

## Colab notebook

For Google Colab setup, troubleshooting, and dependency issues, **see [`Colab.md`](Colab.md)**.  
Quick start: https://colab.research.google.com/github/i221330/LiveCrop/blob/main/notebooks/irrigation_rl.ipynb

## Common commands

```bash
# Install (editable) with all dev+train extras
pip install -r requirements.txt
pip install -e ".[train,plot,dev]"

# Run tests
python3 -m pytest                 # full suite
python3 -m pytest -k test_seed    # single test by keyword

# Generate baseline results + plots
python3 -m agents.baselines --episodes 20

# Train an RL agent (algo: dqn | ppo | a2c)
python3 -m agents.train --algo ppo --timesteps 200000 --seed 42

# Evaluate all trained models vs baselines (produces plots + CSV)
python3 -m agents.evaluate --algos ppo dqn a2c

# Dump a synthetic weather season as CSV
python3 -m irrigation_env.weather --seed 0 --out /tmp/season.csv
```

No linter/formatter is configured — keep style consistent with existing files.

## Architecture

The env is a thin orchestrator around three physics-style modules:

```
irrigation_env/
  weather.py      monthly climatology -> AR(1) temps, Bernoulli rain, Hargreaves ET0
  dynamics.py     FAO-56 soil water balance + FAO-33 yield aggregation (per-zone, vectorized)
  environment.py  IrrigationEnv(gym.Env): wires weather + dynamics, does obs normalization,
                  decodes Discrete(256) -> 4 zones x 4 levels, applies daily-budget scaling,
                  and emits the per-step reward + terminal yield bonus
  __init__.py     registers Irrigation-v0 on import; do NOT use gym.register elsewhere
```

Key design decisions that aren't obvious from the code:

- **Action space is flat `Discrete(256)`, not `MultiDiscrete([4]*4)`.** This keeps DQN viable without wrappers; PPO/A2C don't suffer. Decoding lives in `environment.decode_action` with zone 0 as the LSB.
- **Daily budget is enforced by proportional scaling inside `step()`**, not by masking the action space. Agents learn budgeting as emergent behavior. If you change this, update `test_budget_cap_enforced`.
- **Observation is clipped to `[0, 1]`.** Normalization constants live as module-level `_NORM_*` in `environment.py`. If you widen the weather range or zone count, update both the norms AND `observation_space` dim computation (currently `num_zones + horizon*4 + 2`).
- **Reward is shaped per-step plus a terminal yield bonus.** Magnitudes live in `configs/env.yaml`, not in code — tune there first.
- **Weather is fully synthetic** (no network calls, no baked CSV). The climatology in `weather._CLIMATE_POINTS` is calibrated to Fresno 1991–2020 normals. If you swap in real NASA POWER data later, keep the same column order: `[precip_mm, tmax_C, tmin_C, et0_mm]`.

Dynamics uses water in **mm of root-zone depth**, not volumetric water content. Conversions live as `@property` methods on `SoilParams` (`.fc_mm`, `.wp_mm`, `.sat_mm`, `.taw_mm`). Never mix units — always stay in mm inside the env.

## Config flow

`configs/env.yaml` is loaded by `load_config()` and hydrated into a frozen `EnvConfig` dataclass. The env constructor accepts either a config object or a path; tests rely on the default path resolving relative to the repo root. When adding a new config field, update `load_config` AND `EnvConfig` AND `configs/env.yaml` in the same commit.

## Testing invariants

`tests/test_env.py` pins the core invariants: Gymnasium API compliance via `check_env`, seed determinism, water-budget cap, physical bounds on soil water, yield ordering between always-off/always-max policies, and piecewise-linear Kc. Any env change should leave these green. Add new invariants here rather than in ad-hoc scripts.

## Reproducibility

All randomness flows through `env.np_random` (set by `gym.Env.reset(seed=...)`). The weather generator takes a `np.random.Generator`, never its own seeds. When adding new stochasticity (e.g. stochastic irrigation failures), source it from `self.np_random` for free reproducibility.

## Results artifacts

- `results/plots/` — **tracked**, these are the portfolio deliverables
- `results/logs/`, `results/models/`, `results/raw/` — gitignored

Don't commit TensorBoard event files or model `.zip`s. Do commit the plots that the README references.

## Roadmap status

Week 1 (env + baselines) is complete. Week 2 is complete: `agents/train.py` (DQN/PPO/A2C via SB3), `agents/evaluate.py` (model vs baseline comparison + trajectory plots), and `notebooks/irrigation_rl.ipynb` (Colab-ready, includes seed sweep section). Next: run the seed sweeps, commit plots, write README training section.
