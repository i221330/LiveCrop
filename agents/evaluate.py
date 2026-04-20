"""Evaluate trained RL models against baselines on Irrigation-v0.

Loads saved SB3 models from results/models/, runs seed sweeps, and
produces a unified comparison CSV + plot covering all policies.

Usage:
    python -m agents.evaluate
    python -m agents.evaluate --algos ppo a2c --eval-seeds 50
    python -m agents.evaluate --algos dqn --seed-model 0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import irrigation_env  # noqa: F401 - registers Irrigation-v0
from agents.baselines import (
    EpisodeResult,
    MoistureThresholdPolicy,
    make_comparison_plot,
    random_policy,
    run_episodes,
    summarize,
)


def run_model_episodes(
    model_path: Path,
    algo: str,
    seeds: List[int],
) -> List[EpisodeResult]:
    """Load a saved SB3 model and run deterministic rollouts."""
    from stable_baselines3 import A2C, DQN, PPO
    from irrigation_env.environment import IrrigationEnv

    cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C}[algo]
    model = cls.load(str(model_path))

    env = IrrigationEnv()
    results: List[EpisodeResult] = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        total_r = 0.0
        last_info: Dict = {}
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            total_r += r
            last_info = info
            if term or trunc:
                break
        results.append(
            EpisodeResult(
                policy=algo,
                seed=seed,
                total_reward=total_r,
                mean_yield=float(last_info.get("mean_yield", 0.0)),
                water_used_mm=float(last_info.get("episode_water_used_mm", 0.0)),
            )
        )
    return results


def make_rl_trajectory_plot(
    algo: str,
    model_path: Path,
    seed: int,
    out_path: Path,
) -> None:
    """Side-by-side season trajectory: threshold heuristic vs trained agent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stable_baselines3 import A2C, DQN, PPO
    from irrigation_env.environment import IrrigationEnv

    cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C}[algo]
    model = cls.load(str(model_path))

    policies = {
        "threshold": MoistureThresholdPolicy(),
        algo: lambda env, obs: int(model.predict(obs, deterministic=True)[0]),
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for name, policy in policies.items():
        env = IrrigationEnv()
        obs, _ = env.reset(seed=seed)
        moist_hist, water_hist, reward_hist = [], [], []
        while True:
            action = policy(env, obs)
            obs, r, term, trunc, info = env.step(action)
            moist_hist.append(info["water_mm"].copy())
            water_hist.append(float(info["water_applied_mm"].sum()))
            reward_hist.append(r)
            if term or trunc:
                break

        import numpy as _np
        moist = _np.stack(moist_hist, axis=0)
        days = _np.arange(moist.shape[0])
        axes[0].plot(days, moist.mean(axis=1), label=name)
        axes[1].plot(days, water_hist, label=name, alpha=0.8)
        axes[2].plot(days, _np.cumsum(reward_hist), label=name)

    cfg = IrrigationEnv().cfg
    axes[0].axhline(cfg.soil.fc_mm, color="gray", ls="--", lw=0.8, label="FC")
    axes[0].axhline(cfg.soil.wp_mm, color="red", ls="--", lw=0.8, label="WP")
    axes[0].set_ylabel("Soil water (mm, mean zones)")
    axes[1].set_ylabel("Daily water applied (mm)")
    axes[2].set_ylabel("Cumulative reward")
    axes[2].set_xlabel("Day of season")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Trajectory: threshold vs {algo.upper()} (seed={seed})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Trajectory plot → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL models vs baselines on Irrigation-v0.")
    parser.add_argument(
        "--algos", nargs="+", choices=["dqn", "ppo", "a2c"],
        default=["dqn", "ppo", "a2c"],
    )
    parser.add_argument("--seed-model", type=int, default=42,
                        help="Training seed used when naming the model file.")
    parser.add_argument("--eval-seeds", type=int, default=30)
    parser.add_argument(
        "--seed-start", type=int, default=2000,
        help="Eval seeds start here (keep distinct from training seeds).",
    )
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    seeds = list(range(args.seed_start, args.seed_start + args.eval_seeds))

    all_results: List[EpisodeResult] = []

    print("Running baselines …")
    all_results.extend(run_episodes("random", random_policy, seeds))
    all_results.extend(run_episodes("threshold", MoistureThresholdPolicy(), seeds))

    found_algos: List[str] = []
    for algo in args.algos:
        model_path = out_dir / "models" / f"{algo}_seed{args.seed_model}.zip"
        if not model_path.exists():
            print(f"  [skip] {model_path} not found — run: python -m agents.train --algo {algo}")
            continue
        print(f"Evaluating {algo.upper()} …")
        # load strips the .zip suffix itself
        all_results.extend(run_model_episodes(model_path.with_suffix(""), algo, seeds))
        found_algos.append(algo)

    summary = summarize(all_results)
    print(f"\nEvaluation summary ({len(seeds)} seeds):")
    for name, s in summary.items():
        print(
            f"  {name:<10s}  reward {s['reward_mean']:7.2f} ± {s['reward_std']:5.2f}   "
            f"yield {s['yield_mean']:.3f} ± {s['yield_std']:.3f}   "
            f"water {s['water_mean_mm']:6.0f} ± {s['water_std_mm']:5.0f} mm"
        )

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / "eval_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "seed", "total_reward", "mean_yield", "water_used_mm"])
        for r in all_results:
            w.writerow([r.policy, r.seed, r.total_reward, r.mean_yield, r.water_used_mm])
    print(f"\nCSV → {csv_path}")

    with open(raw_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON → {raw_dir / 'eval_summary.json'}")

    if not args.skip_plots:
        plot_path = out_dir / "plots" / "eval_comparison.png"
        make_comparison_plot(all_results, plot_path)
        print(f"Comparison plot → {plot_path}")

        for algo in found_algos:
            model_path = out_dir / "models" / f"{algo}_seed{args.seed_model}"
            traj_path = out_dir / "plots" / f"trajectory_{algo}.png"
            make_rl_trajectory_plot(algo, model_path, seed=seeds[0], out_path=traj_path)


if __name__ == "__main__":
    main()
