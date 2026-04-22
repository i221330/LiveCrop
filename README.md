# Irrigation Scheduling RL

A custom Gymnasium environment for irrigation scheduling, benchmarked across **DQN, PPO, and A2C** using Stable-Baselines3.

> **Status:** Week 1–3 complete. Environment, baselines, and RL training pipeline (DQN/PPO/A2C) with Colab notebook.

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

## Results

**Comparison (Week 3):** DQN, PPO, and A2C trained for 200k steps each; evaluated on 30 held-out seeds.

![Comparison boxplot](results/plots/eval_comparison.png)

![PPO vs threshold trajectory](results/plots/trajectory_ppo.png)

To train and evaluate:
- **Colab (recommended):** https://colab.research.google.com/github/i221330/LiveCrop/blob/main/notebooks/irrigation_rl.ipynb
- **Local:** `python3 -m agents.train --algo ppo --timesteps 200000 && python3 -m agents.evaluate --algos ppo dqn a2c`

## Roadmap

- [x] Week 1 — custom env, dynamics, tests, random + heuristic baselines
- [x] Week 2 — DQN/PPO/A2C training scripts, Colab notebook, evaluation pipeline
- [x] Week 3 — train all agents, generate comparison plots, push from Colab
- [ ] Week 4 — seed sweeps, hyperparameter tuning, final README polish

## License

MIT
