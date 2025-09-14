import json
import os
from datetime import datetime
from typing import Any

from grid2op.Reward import BaseReward, EpisodeDurationReward
from l2rpn_baselines.PPO_SB3.utils import SB3Agent, save_used_attribute
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.ppo.policies import ActorCriticPolicy

from ppo_on_grid2op.env_utils import make_discrete_action_gym_env
from ppo_on_grid2op.utils import read_config, use_cuda


def train_topological_ppo(
    env_name: str,
    iterations: int,
    reward: type[BaseReward] = EpisodeDurationReward,
    model_policy: type[ActorCriticPolicy] = MlpPolicy,
    safe_max_rho: float = 0.99,
    net_hyperparameters: dict[str, Any] = {
        "net_arch": [100, 100, 100],
    },
    agent_hyperparameters: dict[str, Any] = {
        "gamma": 0.99,
        "n_steps": 16,
        "batch_size": 16,
        "learning_rate": 3e-6,
    },
    callbacks: list[BaseCallback] | None = None,
    prefix_folder: str | None = None,
    model_name_suffix: str | None = None,
    verbose: int = 0,
) -> tuple[SB3Agent, str]:
    """Train a PPO agent with only topological actions.
    Slight modification and refactoring of https://github.com/Grid2op/l2rpn-baselines/blob/master/l2rpn_baselines/PPO_SB3/train.py
    Main difference is in the ActionSpace which in our case is DiscreteActionSpace and not BoxActionSpace
    Args:
        env_name (str): environment to train the agent on
        iterations (int): number of training iterations
        reward (type[BaseReward]): what reward to use during training
        model_policy (type[ActorCriticPolicy]): policy class to use
            Defaults to MlpPolicy which is the same default of L2RPN baselines
        safe_max_rho (float): thermal limit over which the agent will take an action to prevent a blackout. Defaults to 0.99.
        learning_rate (float): PPO optimization step size. Defaults to 3e-6
        hyperparameters (dict[str, Any]): PPO hyperparameters.
            Defaults to
            {
            "net_arch": [100, 100, 100],
            "gamma": 0.99,
            "n_steps": 16,
            "batch_size": 16,
            }
        callbacks (list[BaseCallback], optional): list of callbacks for agent learning. Defaults to None.
        prefix_folder (str, optional): subfolder to create in  model folder, useful for hyperparameter tuning. Defaults to None.
        model_name_suffix (str, optional): name suffix to track specific runs, useful for tuning. Defaults to None, no suffix.
        verbose (int): verbosity level from 0 onwards. Defaults to 0.
    Returns:
        tuple[SB3Agent, str]: trained agent and agent name
    """
    config = read_config()
    set_random_seed(config["seed"], using_cuda=use_cuda())
    timestamp = "".join(str(datetime.now()).split(".")[0:-1]).replace(" ", "_")

    model_parent_folder = (
        config["models_dir"]
        if prefix_folder is None
        else os.path.join(config["models_dir"], prefix_folder)
    )
    model_name = f"PPO_{model_name_suffix + '_' if model_name_suffix is not None else ''}env={env_name}_iterations={iterations}_{timestamp}"
    model_path = os.path.join(model_parent_folder, model_name)

    gymenv_kwargs = {"safe_max_rho": safe_max_rho}
    env_gym, env = make_discrete_action_gym_env(
        env_name,
        "train",
        config["obs_features"],
        config["selected_actions"],
        reward,
        gymenv_kwargs,  # type: ignore[bad-argument-type]
        config["chronics_filter"],
        seed=config["seed"],
        validation_set_percentage=config["validation_set_percentage"],
        test_set_percentage=config["test_set_percentage"],
    )

    save_used_attribute(
        model_parent_folder,
        model_name,
        config["obs_features"],
        config["selected_actions"],
    )

    with open(
        os.path.join(model_path, ".normalize_obs"), encoding="utf-8", mode="w"
    ) as f:
        f.write("I have encoded the observation space !\n DO NOT MODIFY !")

    tensorboard_logs_dir = (
        config["tensorboard_logs_dir"]
        if prefix_folder is None
        else os.path.join(config["tensorboard_logs_dir"], prefix_folder)
    )
    nn_kwargs = {
        "policy": MlpPolicy,
        "env": env_gym,
        "verbose": verbose,
        "tensorboard_log": os.path.join(tensorboard_logs_dir, model_name),
        "device": "cuda" if use_cuda() else "cpu",
        "policy_kwargs": net_hyperparameters,
        **agent_hyperparameters,
    }

    agent = SB3Agent(
        env.action_space,
        env_gym.action_space,
        env_gym.observation_space,
        nn_kwargs=nn_kwargs,
    )
    agent.seed(config["seed"])

    # Added to retain heuristics parameters, seed and chronics filtering
    with open(
        os.path.join(model_path, "extra_configurations.json"),
        encoding="utf-8",
        mode="w",
    ) as f:
        extra_configurations_dict = {
            "gymenv_kwargs": gymenv_kwargs,
            "chronics_filter": config["chronics_filter"],
            "seed": config["seed"],
        }
        f.write(json.dumps(extra_configurations_dict))

    checkpoint_callback = CheckpointCallback(
        save_freq=min(
            max(iterations // 10, 100), 1_000_000
        ),  # min save freq updated to 100 for less I/O time
        save_path=model_path,
        name_prefix=model_name,
    )

    agent.nn_model.learn(  # type: ignore
        total_timesteps=iterations,
        callback=[checkpoint_callback] + ([] if callbacks is None else callbacks),
        progress_bar=True if verbose > 0 else False,
    )

    agent.nn_model.save(os.path.join(model_path, model_name))  # type: ignore[possibly-unbound-attribute]

    env_gym.close()
    return agent, model_name
