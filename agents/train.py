"""Train DQN, PPO, or A2C on Irrigation-v0 using Stable-Baselines3.

Usage:
    python -m agents.train --algo ppo --timesteps 200000 --seed 42
    python -m agents.train --algo dqn --timesteps 300000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

import irrigation_env  # noqa: F401 - registers Irrigation-v0

# Per-algorithm hyperparameter defaults tuned for Irrigation-v0.
# Episodes are exactly 150 steps, so on-policy n_steps aligns to one season.
_ALGO_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "dqn": {
        "learning_rate": 1e-4,
        "buffer_size": 50_000,
        "learning_starts": 1_500,
        "batch_size": 64,
        "gamma": 0.99,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "train_freq": 4,
        "gradient_steps": 1,
    },
    "ppo": {
        "learning_rate": 3e-4,
        "n_steps": 150,
        "batch_size": 150,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "clip_range": 0.2,
    },
    "a2c": {
        "learning_rate": 7e-4,
        "n_steps": 150,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
    },
}


def make_model(algo: str, env, log_dir: Path, seed: int):
    from stable_baselines3 import A2C, DQN, PPO

    cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C}[algo]
    return cls(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        tensorboard_log=str(log_dir),
        verbose=0,
        **_ALGO_DEFAULTS[algo],
    )


def eval_model(model, n_episodes: int = 20, seed_start: int = 1000) -> Dict[str, float]:
    """Run deterministic rollouts on a fresh env; returns summary stats."""
    from irrigation_env.environment import IrrigationEnv

    env = IrrigationEnv()
    rewards, yields, waters = [], [], []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_start + i)
        total_r = 0.0
        last_info: Dict = {}
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            total_r += r
            last_info = info
            if term or trunc:
                break
        rewards.append(total_r)
        yields.append(float(last_info.get("mean_yield", 0.0)))
        waters.append(float(last_info.get("episode_water_used_mm", 0.0)))

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "yield_mean": float(np.mean(yields)),
        "yield_std": float(np.std(yields)),
        "water_mean_mm": float(np.mean(waters)),
        "water_std_mm": float(np.std(waters)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL agent on Irrigation-v0.")
    parser.add_argument("--algo", choices=["dqn", "ppo", "a2c"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    from stable_baselines3.common.env_util import make_vec_env
    from irrigation_env.environment import IrrigationEnv

    out_dir = Path(args.out_dir)
    run_name = f"{args.algo}_seed{args.seed}"
    log_dir = out_dir / "logs" / run_name
    model_dir = out_dir / "models"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(IrrigationEnv, n_envs=1, seed=args.seed)
    model = make_model(args.algo, env, log_dir, args.seed)

    print(f"Training {args.algo.upper()} for {args.timesteps:,} timesteps  (seed={args.seed}) …")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, tb_log_name=run_name, progress_bar=True)
    print(f"Done in {time.time() - t0:.1f}s")

    model_path = model_dir / run_name
    model.save(str(model_path))
    print(f"Model saved → {model_path}.zip")

    print(f"Evaluating over {args.eval_episodes} held-out episodes …")
    stats = eval_model(model, n_episodes=args.eval_episodes)
    stats.update({"algo": args.algo, "seed": args.seed, "timesteps": args.timesteps})

    summary_path = out_dir / "raw" / f"train_{run_name}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(
        f"\nEval ({args.eval_episodes} episodes):  "
        f"reward {stats['reward_mean']:.2f} ± {stats['reward_std']:.2f}   "
        f"yield {stats['yield_mean']:.3f} ± {stats['yield_std']:.3f}   "
        f"water {stats['water_mean_mm']:.0f} ± {stats['water_std_mm']:.0f} mm"
    )
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
