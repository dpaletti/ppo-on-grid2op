import logging
import re
from functools import partial
from typing import Literal

import grid2op
from grid2op.Chronics import Multifolder, MultifolderWithCache
from grid2op.Environment import Environment
from grid2op.Exceptions import UnknownEnv
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace, GymEnv
from grid2op.Reward import BaseReward
from lightsim2grid import LightSimBackend  # type: ignore[possibly-unbound-import]

from ppo_on_grid2op.env_with_heuristics import GymEnvWithRecoWithDNWithShuffle
from ppo_on_grid2op.graph_gym_obs_space import GraphGymObsSpace
from ppo_on_grid2op.masked_env import MaskedGymEnvWithRecoWithDNWithShuffle


def make_discrete_action_gym_env(
    env_name: str,
    suffix: Literal["train", "val", "test"],
    obs_features: list[str],
    selected_actions: list[str],
    reward_class: type[BaseReward],
    gymenv_kwargs: dict[str, int | float | str],
    chronics_filter: str,
    seed: int,
    validation_set_percentage: float | None = None,
    test_set_percentage: float | None = None,
    disable_cache: bool = False,
    disable_shuffle: bool = False,
    enable_masking: bool = False,
    enable_graph: bool = False,
) -> tuple[GymEnv, Environment]:
    """Create environment with 'obs_features' in a BoxSpace encoding and 'selected_actions' in a DiscreteSpace.
    Closely resemble the environment building in https://github.com/Grid2op/l2rpn-baselines/blob/master/l2rpn_baselines/PPO_SB3/train.py
    Modifies the action encoding from  Box to Discrete which is more suitable for purely topological actions.

    Args:
        env_name (str): grid2op environment name
        suffix (Literal["train", "val", "test"]): suffix to add to env name (e.g. train, val, test...)
        obs_features (list[str]): observation features to take into account.
        In purely topological task ignore dispatching and curtailment features.
        selected_actions (list[str]): actions available to the agent
        reward_class (type[BaseReward]): what reward to use in the given environment
        chronics_filter (str): chronics representing which
        seed (int): random seed
        validation_set_percentage (float, optional):
            a value between 1 and 100 to split the environment. If the env has already been split gets ignored.
            Defaults to None.
        test_set_percentage (float, optional):
            a value between 1 and 100 to split the environment. If the env has already been split gets ignored.
            Defaults to None.
        disable_cache (bool): whether to disable caching in grid2op environment
        disable_shuffle (bool): whether to disable periodic chronics shuffling, useful for evaluation
        enable_masking (bool): whether to enable action masking. Defaults to False.
        enable_graph (bool): whether to enable graph observations. Defaults to False
    Raises:
        ValueError: if validation_set_percentage or test_set_percentage is None and the env has not already been split
    """
    try:
        env = make_seed_and_preload_grid2op_env(
            env_name + f"_{suffix}", reward_class, chronics_filter, seed, disable_cache
        )
    except UnknownEnv:
        logging.info(
            f"Could not find environment {env_name}_{suffix} trying to split environment {env_name} in train, validation, test"
        )
        if validation_set_percentage is None or test_set_percentage is None:
            raise ValueError(
                f"Found 'validation_set_percentage'={validation_set_percentage} and test_set_percentage={test_set_percentage}."
                "Both values must be not None for the env to be created correctly, alternatively split the environment before calling this function."
            )
        env = make_seed_and_preload_grid2op_env(
            env_name, reward_class, chronics_filter, seed, disable_cache
        )
        env.train_val_split_random(
            pct_val=validation_set_percentage,
            pct_test=test_set_percentage,
            add_for_test="test",
        )
        logging.info(
            "Environment split in train validation and test, next time the split won't be repeated."
            " Training set has suffix _train, validation has _val and test has _test"
        )

        env = make_seed_and_preload_grid2op_env(
            env_name + f"_{suffix}", reward_class, chronics_filter, seed, disable_cache
        )

    env_gym = (
        GymEnvWithRecoWithDNWithShuffle(
            env, disable_shuffle=disable_shuffle, **gymenv_kwargs
        )
        if not enable_masking
        else MaskedGymEnvWithRecoWithDNWithShuffle(
            env_init=env, disable_shuffle=disable_shuffle, **gymenv_kwargs
        )
    )

    if not enable_graph:
        env_gym.observation_space = BoxGymObsSpace(
            env.observation_space,
            attr_to_keep=obs_features,
        )
        for obs_feature in obs_features:
            env_gym.observation_space.normalize_attr(obs_feature)
    else:
        graph = env.reset().get_energy_graph()
        env_gym.observation_space = GraphGymObsSpace(
            grid2op_observation_space=env.observation_space,
            attr_to_keep=obs_features,
            n_nodes=env.n_sub * env.n_busbar_per_sub,
            n_edges=env.n_line,
            node_feature_space_size=len(graph.nodes[0]),
            edge_feature_space_size=len(graph.edges[[e for e in graph.edges][0]]),
        )  # already normalized

    env_gym.action_space = DiscreteActSpace(
        env.action_space, attr_to_keep=selected_actions
    )

    return env_gym, env


def make_seed_and_preload_grid2op_env(
    env_name: str,
    reward_class: type[BaseReward],
    chronics_filter: str,
    seed: int,
    disable_cache: bool = False,
) -> Environment:
    """Make grid2op env with seeding and MultifolderWithCache support.
        LightSimBackend is also employed for better performance.
        Chronics preloaded into memory are filtered with the chronics filter
    Args:
        env_name (str): env to create
        reward_class (type[BaseReward]): what reward to use when creating the environment.
        chronics_filter (str): regex matching all the chronics to retain
        seed (int): random generation seed for the environment
        disable_cache (bool): whether to use MultiFolder or MultiFolderWithCache, needed for evaluation that does not support caching.

    Returns:
        Environment: grid2op environment
    """
    env = grid2op.make(
        env_name,
        backend=LightSimBackend(),
        chronics_class=MultifolderWithCache if not disable_cache else Multifolder,
        reward_class=reward_class,
    )
    env.chronics_handler.real_data.set_filter(  # type: ignore
        partial(_match_chronics_filter, chronics_filter)
    )
    env.chronics_handler.real_data.reset()  # type: ignore

    env.seed(seed)  # seeding must happen after warming up cache
    env.reset()  # needed to make seeding work
    return env


def _match_chronics_filter(pattern: str, chronics: str) -> bool:
    """Workaround function as lambdas cannot be pickled and grid2op runner needs pickling.

    Args:
        pattern (str): pattern to match
        chronics (str): chronic to match against

    Returns:
        bool: True if chronic has been matched False otherwise
    """
    return re.match(pattern, chronics) is not None
