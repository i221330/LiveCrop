"""LiveCrop — Streamlit dashboard.

Three tabs:
  Simulate  — run any policy for a chosen seed, interactive Plotly season view
  Compare   — portfolio plots + eval summary table
  Tune      — reward-weight sliders, re-run threshold policy live
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import streamlit as st

import irrigation_env  # noqa: F401 - registers Irrigation-v0
from irrigation_env.environment import EnvConfig, IrrigationEnv, load_config
from agents.baselines import MoistureThresholdPolicy, random_policy

RESULTS = Path("results")
MODELS_DIR = RESULTS / "models"

st.set_page_config(
    page_title="LiveCrop — Irrigation RL",
    page_icon="🌱",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(p.stem for p in MODELS_DIR.glob("*.zip"))


@st.cache_data(show_spinner=False)
def run_episode(policy_name: str, seed: int, cfg_overrides: Optional[Dict] = None):
    """Run a full episode and return step-by-step history dict."""
    if cfg_overrides:
        import yaml, copy
        from irrigation_env.environment import DEFAULT_CONFIG_PATH
        from irrigation_env.dynamics import SoilParams
        with open(DEFAULT_CONFIG_PATH) as f:
            raw = yaml.safe_load(f)
        raw["reward"].update(cfg_overrides.get("reward", {}))
        if "daily_budget_mm" in cfg_overrides:
            raw["action"]["daily_budget_mm"] = cfg_overrides["daily_budget_mm"]
        env = IrrigationEnv(config=load_config.__wrapped__(raw) if hasattr(load_config, "__wrapped__") else _cfg_from_raw(raw))
    else:
        env = IrrigationEnv()

    if policy_name == "random":
        policy_fn = random_policy
    elif policy_name == "threshold":
        policy_fn = MoistureThresholdPolicy()
    else:
        policy_fn = _load_model_policy(policy_name)

    obs, _ = env.reset(seed=seed)
    days, moisture, water_applied, rain, et0, kc, rewards, cum = [], [], [], [], [], [], [], []
    total_r = 0.0
    while True:
        action = policy_fn(env, obs)
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        days.append(info["day"])
        moisture.append(info["water_mm"].copy())
        water_applied.append(info["water_applied_mm"].copy())
        rain.append(float(info["rain_mm"]))
        et0.append(float(info["et0_mm"]))
        kc.append(float(info["kc"]))
        rewards.append(r)
        cum.append(total_r)
        if term or trunc:
            break

    return {
        "days": days,
        "moisture": np.array(moisture),
        "water_applied": np.array(water_applied),
        "rain": rain,
        "et0": et0,
        "kc": kc,
        "rewards": rewards,
        "cumulative": cum,
        "final_info": info,
        "cfg": env.cfg,
    }


def _load_model_policy(model_name: str):
    """Return a callable policy from a saved SB3 model."""
    algo = model_name.split("_")[0]
    from stable_baselines3 import A2C, DQN, PPO
    cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C}[algo]
    model = cls.load(str(MODELS_DIR / model_name))
    def policy(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return policy


def _cfg_from_raw(raw: dict) -> EnvConfig:
    """Re-hydrate an EnvConfig from a modified raw YAML dict."""
    from irrigation_env.dynamics import SoilParams
    z = raw["zones"]
    soil = SoilParams(
        root_zone_depth_mm=float(z["root_zone_depth_mm"]),
        field_capacity_theta=float(z["field_capacity_theta"]),
        wilting_point_theta=float(z["wilting_point_theta"]),
        saturation_theta=float(z["saturation_theta"]),
    )
    return EnvConfig(
        num_zones=int(z["num_zones"]),
        season_length=int(raw["season"]["length_days"]),
        start_month=int(raw["season"]["start_month"]),
        water_levels_mm=np.array(raw["action"]["water_levels_mm"], dtype=np.float32),
        daily_budget_mm=float(raw["action"]["daily_budget_mm"]),
        forecast_horizon=int(raw["forecast"]["horizon_days"]),
        soil=soil,
        initial_moisture_theta=float(z["initial_moisture_theta"]),
        zone_moisture_noise_fraction=float(z.get("zone_moisture_noise_fraction", 0.0)),
        kc_stages=[tuple(s) for s in raw["crop"]["kc_stages"]],
        ky_sensitivity=float(raw["crop"]["ky_sensitivity"]),
        water_cost_per_mm=float(raw["reward"]["water_cost_per_mm"]),
        stress_penalty_weight=float(raw["reward"]["stress_penalty_weight"]),
        waterlogging_penalty_weight=float(raw["reward"]["waterlogging_penalty_weight"]),
        yield_bonus_weight=float(raw["reward"]["yield_bonus_weight"]),
    )


def _season_figure(h: dict):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    cfg = h["cfg"]
    days = h["days"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Soil moisture per zone (mm)",
            "Daily water applied per zone (mm)",
            "Rain & ET₀ (mm)",
            "Cumulative reward",
        ],
        vertical_spacing=0.07,
    )

    # Moisture
    for z in range(cfg.num_zones):
        fig.add_trace(go.Scatter(
            x=days, y=h["moisture"][:, z],
            name=f"Zone {z+1}", line=dict(color=colors[z]),
            legendgroup="zone", legendgrouptitle_text="Zones" if z == 0 else "",
        ), row=1, col=1)
    fig.add_hline(y=cfg.soil.fc_mm, line_dash="dash", line_color="gray",
                  annotation_text="FC", annotation_position="right", row=1, col=1)
    fig.add_hline(y=cfg.soil.wp_mm, line_dash="dot", line_color="red",
                  annotation_text="WP", annotation_position="right", row=1, col=1)

    # Water applied
    for z in range(cfg.num_zones):
        fig.add_trace(go.Bar(
            x=days, y=h["water_applied"][:, z],
            name=f"Zone {z+1}", marker_color=colors[z],
            legendgroup="zone", showlegend=False,
        ), row=2, col=1)

    # Rain + ET0
    fig.add_trace(go.Bar(x=days, y=h["rain"], name="Rain", marker_color="#64B5F6",
                         legendgroup="weather"), row=3, col=1)
    fig.add_trace(go.Scatter(x=days, y=h["et0"], name="ET₀", line=dict(color="#FF7043"),
                             legendgroup="weather"), row=3, col=1)

    # Cumulative reward
    fig.add_trace(go.Scatter(
        x=days, y=h["cumulative"],
        name="Reward", fill="tozeroy",
        line=dict(color="#9C27B0"), showlegend=False,
    ), row=4, col=1)

    fig.update_layout(
        height=860, barmode="stack",
        legend=dict(orientation="h", y=-0.06, x=0),
        margin=dict(t=40, b=20),
    )
    fig.update_xaxes(title_text="Day of season", row=4)
    fig.update_yaxes(title_text="mm", row=1)
    fig.update_yaxes(title_text="mm", row=2)
    fig.update_yaxes(title_text="mm", row=3)
    return fig


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("🌱 LiveCrop — Irrigation Scheduling RL")
st.caption(
    "Custom Gymnasium environment (Irrigation-v0) · "
    "DQN / PPO / A2C via Stable-Baselines3 · "
    "[GitHub](https://github.com/i221330/LiveCrop)"
)

tab_sim, tab_cmp, tab_tune = st.tabs(["🎬 Simulate", "📊 Compare", "⚙️ Tune"])

# ---------------------------------------------------------------------------
# Tab 1 — Simulate
# ---------------------------------------------------------------------------

with tab_sim:
    st.subheader("Run a season with any policy")

    col_ctrl, col_info = st.columns([1, 2])
    with col_ctrl:
        model_names = _find_models()
        policy_options = ["threshold", "random"] + model_names
        policy_labels = {
            "threshold": "Moisture-threshold heuristic",
            "random": "Random (baseline floor)",
            **{m: f"Trained: {m}" for m in model_names},
        }
        policy = st.selectbox(
            "Policy",
            policy_options,
            format_func=lambda k: policy_labels[k],
        )
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
        run_btn = st.button("▶ Run season", width="stretch", type="primary")

    if run_btn or "sim_history" in st.session_state:
        if run_btn:
            with st.spinner("Simulating 150-day season …"):
                st.session_state["sim_history"] = run_episode(policy, seed)
                st.session_state["sim_policy"] = policy

        h = st.session_state["sim_history"]
        cfg = h["cfg"]
        fi = h["final_info"]

        with col_info:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total reward",     f"{h['cumulative'][-1]:.1f}")
            m2.metric("Mean yield",       f"{fi.get('mean_yield', 0):.3f}")
            m3.metric("Water used (mm)",  f"{fi['episode_water_used_mm']:.0f}")

        st.plotly_chart(_season_figure(h), width="stretch")
    else:
        st.info("Choose a policy and hit **▶ Run season** to simulate a full growing season.")

# ---------------------------------------------------------------------------
# Tab 2 — Compare
# ---------------------------------------------------------------------------

with tab_cmp:
    st.subheader("Results — RL agents vs baselines")

    # Summary table
    summary_path = RESULTS / "raw" / "eval_summary.json"
    if summary_path.exists():
        import json, pandas as pd
        with open(summary_path) as f:
            summary = json.load(f)
        rows = []
        for name, s in summary.items():
            rows.append({
                "Policy": name,
                "Reward mean": f"{s['reward_mean']:.2f}",
                "Reward std":  f"{s['reward_std']:.2f}",
                "Yield mean":  f"{s['yield_mean']:.3f}",
                "Yield std":   f"{s['yield_std']:.3f}",
                "Water (mm)":  f"{s['water_mean_mm']:.0f}",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Policy"), width="stretch")
    else:
        st.info("Run `python3 -m agents.evaluate` locally or Section 4 in the Colab notebook to generate `results/raw/eval_summary.json`.")

    st.divider()

    plot_files = {
        "eval_comparison.png": "Comparison — all policies (30 eval seeds)",
        "trajectory_ppo.png":  "Season trajectory — PPO vs threshold",
        "baselines.png":       "Baselines only",
        "baselines_trajectory.png": "Baseline season trajectory",
        "sweep_ppo.png":       "PPO seed sweep — learning curve ribbon",
    }

    found = [k for k in plot_files if (RESULTS / "plots" / k).exists()]
    if found:
        cols = st.columns(min(2, len(found)))
        for i, fname in enumerate(found):
            with cols[i % 2]:
                st.caption(plot_files[fname])
                st.image(str(RESULTS / "plots" / fname), width="stretch")
    else:
        st.warning("No plots found in `results/plots/`. Run the Colab notebook or `agents/evaluate.py` first.")

# ---------------------------------------------------------------------------
# Tab 3 — Tune
# ---------------------------------------------------------------------------

with tab_tune:
    st.subheader("Config explorer — tune reward weights and rerun")
    st.caption("Modifies a copy of `configs/env.yaml` in memory only. Changes are not saved to disk.")

    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown("**Reward weights**")
        water_cost   = st.slider("Water cost per mm",          0.0,  0.1,  0.01, 0.005, format="%.3f")
        stress_w     = st.slider("Stress penalty weight",      0.0,  5.0,  1.0,  0.1)
        waterlog_w   = st.slider("Waterlogging penalty weight", 0.0, 5.0,  1.0,  0.1)
        yield_w      = st.slider("Yield bonus weight",         0.0, 30.0, 10.0,  0.5)
        st.markdown("**Action budget**")
        budget       = st.slider("Daily water budget (mm)",   20.0, 200.0, 80.0, 5.0)
        tune_seed    = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1, key="tune_seed")
        tune_btn     = st.button("▶ Run with these settings", width="stretch", type="primary")

    if tune_btn or "tune_history" in st.session_state:
        overrides = {
            "reward": {
                "water_cost_per_mm":           water_cost,
                "stress_penalty_weight":        stress_w,
                "waterlogging_penalty_weight":  waterlog_w,
                "yield_bonus_weight":           yield_w,
            },
            "daily_budget_mm": budget,
        }
        if tune_btn:
            with st.spinner("Running threshold policy with custom config …"):
                st.session_state["tune_history"] = run_episode("threshold", int(tune_seed), cfg_overrides=overrides)

        h = st.session_state["tune_history"]
        fi = h["final_info"]

        with col_r:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total reward",    f"{h['cumulative'][-1]:.1f}")
            m2.metric("Mean yield",      f"{fi.get('mean_yield', 0):.3f}")
            m3.metric("Water used (mm)", f"{fi['episode_water_used_mm']:.0f}")
            st.plotly_chart(_season_figure(h), width="stretch")
    else:
        with col_r:
            st.info("Adjust sliders and hit **▶ Run with these settings** to see the effect on the threshold policy.")
