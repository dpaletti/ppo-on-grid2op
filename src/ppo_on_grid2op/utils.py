import importlib.resources
import logging
import tomllib
from typing import Any

import grid2op
import torch.cuda
from grid2op.Action import TopologySetAction
from grid2op.Chronics import MultifolderWithCache
from grid2op.gym_compat import GymEnv
from grid2op.Reward import BaseReward
from lightsim2grid import LightSimBackend  # type: ignore[possibly-unbound-import]

import ppo_on_grid2op.resources


def make_env(
    env_name: str,
    reward: type[BaseReward],
    acts_to_keep: list[str],
    obs_features_to_keep: list[str],
    chronics_filter: str = ".*",
):
    """Create gym environment from a grid2op one.


    Args:
        env_name (str): grid2op env name
        reward (BaseReward): reward class to use.
        acts_to_keep (list[str]): list of actions available to the agent.
        obs_features_to_keep (list[str]): list of observation features available to the agent.
        chronics_filter (str, optional):
        filter on the chronics to keep in memory, for small environments no need to set.
        Defaults to ".*" which means no filter.

    Returns:
        _type_: _description_
    """
    env = grid2op.make(
        env_name,
        backend=LightSimBackend(),
        chronics_class=MultifolderWithCache,
        reward_class=reward,
        action_class=TopologySetAction,
    )

    gym_env = GymEnv(env)
    return gym_env


def read_config() -> dict[str, Any]:
    return tomllib.loads(
        importlib.resources.read_text(ppo_on_grid2op.resources, "config.toml")
    )


def use_cuda() -> bool:
    config = read_config()
    if config["use_cuda"] and not torch.cuda.is_available():
        logging.warning(
            "use_cuda set to true but torch cannot detect cuda hardware. CUDA not being used."
        )
        return False
    return True
