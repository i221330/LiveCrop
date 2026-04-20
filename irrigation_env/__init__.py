from gymnasium.envs.registration import register

from irrigation_env.environment import EnvConfig, IrrigationEnv, load_config

register(
    id="Irrigation-v0",
    entry_point="irrigation_env.environment:IrrigationEnv",
    max_episode_steps=150,
)

__all__ = ["IrrigationEnv", "EnvConfig", "load_config"]
