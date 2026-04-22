"""Seed sweep: train one algorithm across N seeds and plot a learning-curve ribbon.

Uses SB3's EvalCallback to checkpoint evaluation reward every `--eval-freq` steps.
After all seeds finish, aligns the checkpoints and plots mean ± 1 std.

Usage:
    python -m agents.sweep --algo ppo --seeds 5 --timesteps 200000
    python -m agents.sweep --algo ppo --seeds 5 --timesteps 200000 --eval-freq 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

import irrigation_env  # noqa: F401 - registers Irrigation-v0
from agents.train import eval_model, make_model


def train_one_seed(
    algo: str,
    seed: int,
    timesteps: int,
    eval_freq: int,
    out_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train one seed; return (timestep_checkpoints, mean_eval_rewards)."""
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from irrigation_env.environment import IrrigationEnv

    run_name = f"{algo}_seed{seed}"
    log_dir = out_dir / "logs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(IrrigationEnv, n_envs=1, seed=seed)
    eval_env  = make_vec_env(IrrigationEnv, n_envs=1, seed=seed + 10_000)

    callback = EvalCallback(
        eval_env,
        n_eval_episodes=10,
        eval_freq=eval_freq,
        log_path=str(log_dir),
        best_model_save_path=None,
        deterministic=True,
        verbose=0,
    )

    model = make_model(algo, train_env, log_dir, seed)
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)

    model.save(str(out_dir / "models" / run_name))

    data = np.load(log_dir / "evaluations.npz")
    # data["results"] shape: (n_evals, n_eval_episodes)
    return data["timesteps"], data["results"].mean(axis=1)


def make_ribbon_plot(
    algo: str,
    all_ts: List[np.ndarray],
    all_rewards: List[np.ndarray],
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use the shortest checkpoint sequence for alignment
    min_len = min(len(ts) for ts in all_ts)
    ts = all_ts[0][:min_len]
    rewards = np.stack([r[:min_len] for r in all_rewards])  # (n_seeds, n_evals)

    mean = rewards.mean(axis=0)
    std  = rewards.std(axis=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts, mean, linewidth=2, label=f"{algo.upper()} mean")
    ax.fill_between(ts, mean - std, mean + std, alpha=0.25, label="± 1 std")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean episode reward")
    ax.set_title(f"{algo.upper()} seed sweep ({len(all_ts)} seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Ribbon plot → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed sweep for one algorithm on Irrigation-v0.")
    parser.add_argument("--algo", choices=["dqn", "ppo", "a2c"], default="ppo")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (0 … seeds-1)")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=5_000,
                        help="Evaluation checkpoint interval in env steps.")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    all_ts: List[np.ndarray] = []
    all_rewards: List[np.ndarray] = []
    final_stats = []

    for seed in range(args.seeds):
        print(f"\n{'='*55}")
        print(f"{args.algo.upper()}  seed={seed}  ({args.timesteps:,} steps)")
        ts, rewards = train_one_seed(
            args.algo, seed, args.timesteps, args.eval_freq, out_dir
        )
        all_ts.append(ts)
        all_rewards.append(rewards)
        final_stats.append({"seed": seed, "final_eval_reward": float(rewards[-1])})
        print(f"Final eval reward: {rewards[-1]:.2f}")

    rewards_arr = np.stack([r[:min(len(r) for r in all_rewards)] for r in all_rewards])
    mean_final = float(rewards_arr[:, -1].mean())
    std_final  = float(rewards_arr[:, -1].std())
    print(f"\n{args.algo.upper()} over {args.seeds} seeds: {mean_final:.2f} ± {std_final:.2f}")

    summary = {
        "algo": args.algo,
        "n_seeds": args.seeds,
        "timesteps": args.timesteps,
        "final_reward_mean": mean_final,
        "final_reward_std": std_final,
        "per_seed": final_stats,
    }
    summary_path = out_dir / "raw" / f"sweep_{args.algo}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")

    ribbon_path = out_dir / "plots" / f"sweep_{args.algo}.png"
    make_ribbon_plot(args.algo, all_ts, all_rewards, ribbon_path)


if __name__ == "__main__":
    main()
