import json
import multiprocessing as mp
import os
from typing import Any

from grid2op.Reward import BaseReward, EpisodeDurationReward
from grid2op.Runner import Runner
from l2rpn_baselines.PPO_SB3.utils import SB3Agent

from ppo_on_grid2op.env_utils import make_discrete_action_gym_env
from ppo_on_grid2op.utils import read_config


def evaluate_topological_ppo(
    env_name: str,
    model: str | SB3Agent,
    n_eval_episodes: int,
    reward: type[BaseReward] = EpisodeDurationReward,
    n_parallel_evaluations: int = -1,
    verbose: bool = True,
    obs_features: list[str] | None = None,
    selected_actions: list[str] | None = None,
    gymenv_kwargs: dict[str, Any] | None = None,
    chronics_filter: str | None = None,
    seed: int | None = None,
) -> list[Any]:
    """Evaluate Topological PPO agent trained with 'train_topological_ppo'

    Args:
        env_name (str): env on which to evaluate the agent
        model (str | SB3Agent): model name of the agent to evaluate or model instance.
        If model instance then obs_features, selected_actions, gymenv_kwargs, chronics_filter and seed must be set.
        If model instance is string and any of those parameters is set it is ignored.
        n_eval_episodes (int): number of episodes to run the evaluation on
        reward (type[BaseReward]): what reward to use during evaluation
        n_parallel_evaluations (int, optional): Number of parallel evaluations.
            Defaults to -1 which means use all the available cores in multiprocessing forkserver mode.
        verbose (bool, optional): whether to have verbose output. Defaults to True.
        obs_features (list[str], optional): observation features. Defaults to None.
        selected_actions (list[str], optional): actions available to the agent. Defaults to None.
        gymenv_kwargs (dict[str, Any], optional): heuristics setting. Defaults to None.
        chronics_filter (str, optional): regex to match chronics to preload. Defaults to None.
        seed (int, optional): random seed. Defaults to None.

    Returns:
        list[Any]:returns the evaluated agent and the results as a list of tuples.

    """
    if isinstance(model, str):
        config = read_config()
        model_path = os.path.join(config["models_dir"], model)
        with open(
            os.path.join(model_path, "obs_attr_to_keep.json"),
            encoding="utf-8",
            mode="r",
        ) as f:
            obs_features = json.load(fp=f)
        with open(
            os.path.join(model_path, "act_attr_to_keep.json"),
            encoding="utf-8",
            mode="r",
        ) as f:
            selected_actions = json.load(fp=f)

        with open(
            os.path.join(model_path, "extra_configurations.json"),
            encoding="utf-8",
            mode="r",
        ) as f:
            extra_configurations = json.load(fp=f)
    else:
        if not (
            obs_features is not None
            and selected_actions is not None
            and gymenv_kwargs is not None
            and chronics_filter is not None
            and seed is not None
        ):
            raise ValueError(
                "When passing a model obs_features, selected_actions, gymenv_kwargs, chronics_filter and seed must be specified."
                " Found at least one None."
            )

    env_gym, env = make_discrete_action_gym_env(
        env_name,
        "val",
        obs_features,
        selected_actions,
        reward,
        extra_configurations["gymenv_kwargs"]
        if gymenv_kwargs is None
        else gymenv_kwargs,
        extra_configurations["chronics_filter"]
        if chronics_filter is None
        else chronics_filter,
        extra_configurations["seed"] if seed is None else seed,
        disable_cache=True,
        disable_shuffle=True,
    )
    if isinstance(model, str):
        grid2op_agent = SB3Agent(
            env.action_space,
            env_gym.action_space,
            env_gym.observation_space,
            nn_path=os.path.join(model_path, model),
            gymenv=env_gym,
            iter_num=None,  # restore the last training iteration
        )
    else:
        if isinstance(model, SB3Agent):
            grid2op_agent = model
        else:
            # Need to save and reload into a SB3Agent
            # Under the SB3Agent there's simply PPO.load, there's a cleaner way for sure.
            model.save("temp.zip")  # type: ignore
            grid2op_agent = SB3Agent(
                env.action_space,
                env_gym.action_space,
                env_gym.observation_space,
                nn_path="temp",
                gymenv=env_gym,
                iter_num=None,  # restore the last training iteration
            )
            os.remove("temp.zip")

    # Build runner
    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose
    runner = Runner(
        **runner_params,
        agentClass=None,
        agentInstance=grid2op_agent,
        mp_context=mp.get_context(
            "fork"
        ),  # other contexts do not work due to pickling issues
    )  # type: ignore [missing-argument]

    # Run the agent on the scenarios
    if isinstance(model, str):
        os.makedirs(config["evaluation_logs_dir"], exist_ok=True)

    res = runner.run(
        path_save=os.path.join(config["evaluation_logs_dir"], model)
        if isinstance(model, str)
        else None,
        nb_episode=n_eval_episodes,
        nb_process=n_parallel_evaluations
        if n_parallel_evaluations > 0
        else min(os.cpu_count(), n_eval_episodes),
        max_iter=-1,  # assess all the steps the agent is capable of doing
        pbar=verbose,
    )

    return res
