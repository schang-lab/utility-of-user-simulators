# Modified from SkyRL (https://github.com/NovaSky-AI/SkyRL) for utility-of-user-simulators; see rl_training/SkyRL/README.md.
"""Registers the internal gym envs."""

from skyrl_gym.envs.registration import register

# User simulators from "Quantifying the Utility of User Simulators for Building Collaborative LLM Assistants".
register(
    id="sftuser",
    entry_point="skyrl_gym.envs.user_simulator.env:SFTUserEnv",
)

register(
    id="rpuser",
    entry_point="skyrl_gym.envs.user_simulator.env:RPUserEnv",
)

register(
    id="rpuser_persona",
    entry_point="skyrl_gym.envs.user_simulator.env:RPUserPersonaEnv",
)
