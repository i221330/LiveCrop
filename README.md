# Irrigation Scheduling RL

A custom Gymnasium environment for irrigation scheduling, benchmarked across **DQN, PPO, and A2C** using Stable-Baselines3.

> **Status:** Week 1 — environment + baselines. Training pipeline coming in Week 2.

## Problem

Agriculture consumes ~70% of global freshwater. An irrigation agent must decide how much water to allocate to each of 4 farm zones every day, balancing crop yield against water cost and waterlogging risk. The agent observes soil moisture, a short weather forecast, and the crop's growth stage, and must learn a seasonal policy under stochastic weather.

The environment is a **simplified FAO-56 water balance** coupled to an FAO-33 yield model, with weather calibrated to Fresno, CA (California Central Valley).

## Quick start

```bash
pip install -r requirements.txt
pip install -e .
pytest
python -m agents.baselines --episodes 20
```

```python
import gymnasium as gym
import irrigation_env  # registers Irrigation-v0

env = gym.make("Irrigation-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Environment spec

| | |
|---|---|
| **Observation** | `Box(18,)` — 4 zone moistures, 3-day weather forecast (rain/tmax/tmin/ET0), crop Kc, season progress |
| **Action** | `Discrete(256)` — 4 zones × 4 water levels `{0, 5, 15, 25}` mm |
| **Episode length** | 150 days (May 1 – Sep 27) |
| **Reward** | `-water_cost - stress - waterlogging` per step, `+yield_bonus` at terminal |

## Repo layout

```
irrigation_env/   # Gymnasium env + dynamics + weather
agents/           # baselines + training scripts
configs/          # env and algorithm YAML configs
tests/            # pytest suite
scripts/          # reproducibility shell scripts
notebooks/        # Colab demo
results/          # logs, models, plots (gitignored except plots/)
```

## Roadmap

- [x] Week 1 — custom env, dynamics, tests, random + heuristic baselines
- [ ] Week 2 — DQN training, Colab notebook
- [ ] Week 3 — PPO, A2C, seed sweeps, hyperparameter tuning
- [ ] Week 4 — final plots, architecture diagram, demo GIF, polish

## License

MIT
