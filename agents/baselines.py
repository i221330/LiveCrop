"""Non-learning baselines for Irrigation-v0.

Provides two reference policies that the RL agents must beat:

- RandomPolicy: uniform over the action space, shows the floor.
- MoistureThresholdPolicy: a domain-inspired heuristic that irrigates a
  zone back toward field capacity when its moisture falls below a
  configurable threshold, and skips irrigation if rain is forecast.

Running this module as a script produces episode-level summary stats
and a comparison plot in results/plots/baselines.png.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

import irrigation_env  # noqa: F401 - registers Irrigation-v0
from irrigation_env.environment import IrrigationEnv

Policy = Callable[[IrrigationEnv, np.ndarray], int]


def random_policy(env: IrrigationEnv, obs: np.ndarray) -> int:
    return int(env.action_space.sample())


@dataclass
class MoistureThresholdPolicy:
    """Refill zones below `threshold_fraction_of_fc` toward field capacity.

    Subtracts tomorrow's forecast rain from the requested amount, then
    snaps to the closest available discrete water level.
    """

    threshold_fraction_of_fc: float = 0.75
    use_forecast: bool = True

    def __call__(self, env: IrrigationEnv, obs: np.ndarray) -> int:
        cfg = env.cfg
        n = cfg.num_zones
        moisture_mm = obs[:n] * cfg.soil.sat_mm
        threshold_mm = self.threshold_fraction_of_fc * cfg.soil.fc_mm

        target_deficit = np.maximum(cfg.soil.fc_mm - moisture_mm, 0.0)
        if self.use_forecast:
            # Next-day rain lives at obs[n] scaled by the same norm the env uses (20mm).
            next_rain_mm = float(obs[n]) * 20.0
            target_deficit = np.maximum(target_deficit - next_rain_mm, 0.0)

        levels = cfg.water_levels_mm
        indices = np.empty(n, dtype=np.int64)
        for z in range(n):
            d = target_deficit[z]
            if moisture_mm[z] >= threshold_mm or d < levels[1] / 2:
                indices[z] = 0
            else:
                indices[z] = int(np.argmin(np.abs(levels - d)))

        action = 0
        base = 1
        for idx in indices:
            action += int(idx) * base
            base *= cfg.num_actions_per_zone
        return action


@dataclass
class EpisodeResult:
    policy: str
    seed: int
    total_reward: float
    mean_yield: float
    water_used_mm: float


def run_episodes(
    policy_name: str, policy: Policy, seeds: List[int]
) -> List[EpisodeResult]:
    env = IrrigationEnv()
    out: List[EpisodeResult] = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        total = 0.0
        last_info: Dict = {}
        while True:
            action = policy(env, obs)
            obs, r, term, trunc, info = env.step(action)
            total += r
            last_info = info
            if term or trunc:
                break
        out.append(
            EpisodeResult(
                policy=policy_name,
                seed=seed,
                total_reward=total,
                mean_yield=float(last_info["mean_yield"]),
                water_used_mm=float(last_info["episode_water_used_mm"]),
            )
        )
    return out


def summarize(results: List[EpisodeResult]) -> Dict[str, Dict[str, float]]:
    by_policy: Dict[str, List[EpisodeResult]] = {}
    for r in results:
        by_policy.setdefault(r.policy, []).append(r)
    summary: Dict[str, Dict[str, float]] = {}
    for name, rs in by_policy.items():
        rewards = np.array([r.total_reward for r in rs])
        yields = np.array([r.mean_yield for r in rs])
        water = np.array([r.water_used_mm for r in rs])
        summary[name] = {
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "yield_mean": float(yields.mean()),
            "yield_std": float(yields.std()),
            "water_mean_mm": float(water.mean()),
            "water_std_mm": float(water.std()),
        }
    return summary


def save_csv(results: List[EpisodeResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "seed", "total_reward", "mean_yield", "water_used_mm"])
        for r in results:
            w.writerow([r.policy, r.seed, r.total_reward, r.mean_yield, r.water_used_mm])


def make_comparison_plot(results: List[EpisodeResult], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_policy: Dict[str, Dict[str, List[float]]] = {}
    for r in results:
        d = by_policy.setdefault(r.policy, {"reward": [], "yield": [], "water": []})
        d["reward"].append(r.total_reward)
        d["yield"].append(r.mean_yield)
        d["water"].append(r.water_used_mm)

    names = list(by_policy.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title, ylabel in zip(
        axes,
        ["reward", "yield", "water"],
        ["Episode reward", "Mean relative yield", "Water used (mm)"],
        ["reward", "yield fraction", "mm"],
    ):
        data = [by_policy[n][key] for n in names]
        ax.boxplot(data, tick_labels=names, showmeans=True)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    fig.suptitle("Irrigation-v0 baselines")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_trajectory_plot(seed: int, out_path: Path) -> None:
    """Plot one season's moisture + water + reward for each policy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = {
        "random": random_policy,
        "threshold": MoistureThresholdPolicy(),
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for name, policy in policies.items():
        env = IrrigationEnv()
        obs, _ = env.reset(seed=seed)
        moist_hist: List[np.ndarray] = []
        water_hist: List[float] = []
        reward_hist: List[float] = []
        while True:
            action = policy(env, obs)
            obs, r, term, trunc, info = env.step(action)
            moist_hist.append(info["water_mm"].copy())
            water_hist.append(float(info["water_applied_mm"].sum()))
            reward_hist.append(r)
            if term or trunc:
                break

        moist = np.stack(moist_hist, axis=0)  # (T, zones)
        days = np.arange(moist.shape[0])
        axes[0].plot(days, moist.mean(axis=1), label=f"{name} (mean W)")
        axes[1].plot(days, water_hist, label=name, alpha=0.7)
        axes[2].plot(days, np.cumsum(reward_hist), label=name)

    cfg = IrrigationEnv().cfg
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].axhline(cfg.soil.fc_mm, color="gray", ls="--", lw=0.8, label="FC")
    axes[0].axhline(cfg.soil.wp_mm, color="red", ls="--", lw=0.8, label="WP")
    axes[0].set_ylabel("Soil water (mm, mean across zones)")
    axes[1].set_ylabel("Daily water applied (mm)")
    axes[2].set_ylabel("Cumulative reward")
    axes[2].set_xlabel("Day of season")
    fig.suptitle(f"Season trajectory (seed={seed})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run baseline policies on Irrigation-v0.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))

    policies: Dict[str, Policy] = {
        "random": random_policy,
        "threshold": MoistureThresholdPolicy(),
    }

    all_results: List[EpisodeResult] = []
    for name, p in policies.items():
        all_results.extend(run_episodes(name, p, seeds))

    out_dir = Path(args.out_dir)
    save_csv(all_results, out_dir / "raw" / "baselines.csv")
    summary = summarize(all_results)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "raw" / "baselines_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Baseline summary (mean ± std over {} seeds):".format(len(seeds)))
    for name, s in summary.items():
        print(
            f"  {name:<10s}  reward {s['reward_mean']:7.2f} ± {s['reward_std']:5.2f}   "
            f"yield {s['yield_mean']:.3f} ± {s['yield_std']:.3f}   "
            f"water {s['water_mean_mm']:6.0f} ± {s['water_std_mm']:5.0f} mm"
        )

    if not args.skip_plots:
        make_comparison_plot(all_results, out_dir / "plots" / "baselines.png")
        make_trajectory_plot(seeds[0], out_dir / "plots" / "baselines_trajectory.png")
        print(f"\nPlots written under {out_dir / 'plots'}/")


if __name__ == "__main__":
    main()
