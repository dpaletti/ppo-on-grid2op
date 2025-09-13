import logging
import os
from datetime import datetime
from functools import partial
from typing import Any

import optuna
from grid2op.Reward import BaseReward, EpisodeDurationReward
from optuna import Trial
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from stable_baselines3.a2c.policies import MlpPolicy
from stable_baselines3.ppo.policies import ActorCriticPolicy

from ppo_on_grid2op.topological_ppo_training import train_topological_ppo
from ppo_on_grid2op.trial_eval_callback import TrialEvalCallback
from ppo_on_grid2op.utils import read_config, read_hyperparameters


def tune_topological_ppo(
    env_name: str,
    n_trials: int,
    model_class: str = "PPO",
    reward_class: type[BaseReward] = EpisodeDurationReward,
    n_sampler_startup_trials: int = 10,
    min_timesteps_before_pruning: int = 1000,
    max_timesteps_per_trial: int = 10000,
    hyperband_reduction_factor: int = 3,
    n_eval_episodes: int = 5,
    eval_freq: int = 1000,
    model_policy: type[ActorCriticPolicy] = MlpPolicy,
    seed: int = 0,
    verbose: int = 0,
) -> dict[str, Any]:
    """Run hyperparameter tuning for a given model type in a given environment.
    Hyperparameter space is read from a config determined by 'model_class'

    Args:
        env_name (str): env on which to run the tuning
        n_trials (int): number of trials to run
        model_class (str, optional): model class that determines which hyperparam space to read. Defaults to "PPO".
        reward_class (type[BaseReward], optional): reward to use for training. Defaults to EpisodeDurationReward.
        n_sampler_startup_trials (int, optional): how many random trials to run before using the samplers. Defaults to 10.
        min_timesteps_before_pruning (int, optional): how many timesteps to wait before evaluating pruning. Defaults to 1000.
        max_timesteps_per_trial (int, optional): how many timesteps use to train each trial. Defaults to 10000.
        hyperband_reduction_factor (int, optional): how aggressive the pruner is (3 = balanced). Defaults to 3.
        n_eval_episodes (int, optional): how many episodes to average to evaluate the models. Defaults to 5.
        eval_freq (int, optional): how frequently to evaluate the. Defaults to 1000.
        model_policy (type[ActorCriticPolicy], optional): Which policy to use for the model. Defaults to MlpPolicy.
        seed (int, optional): random generators seed. Defaults to 0.
        verbose (int, optional): how much to log, the higher the value the more logs, 0 means as little as possible. Defaults to 0.

    Returns:
        dict[str, Any]: best parameter set
    """
    # setting multivariate TPESampler for possible correlation between learning rate and batch size
    sampler = TPESampler(
        n_startup_trials=n_sampler_startup_trials, seed=seed, multivariate=True
    )
    pruner = HyperbandPruner(
        min_resource=min_timesteps_before_pruning,
        max_resource=max_timesteps_per_trial,
        reduction_factor=hyperband_reduction_factor,
    )
    # Create the study and start the hyperparameter optimization
    hparam_space = read_hyperparameters(model_class)
    config = read_config()

    timestamp = "".join(str(datetime.now()).split(".")[0:-1]).replace(" ", "_")

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

    study.optimize(
        partial(
            objective,
            env_name=env_name,
            reward_class=reward_class,
            config=config,
            hparam_space=hparam_space,
            max_training_iterations=max_timesteps_per_trial,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            model_policy=model_policy,
            verbose=verbose,
        ),
        n_trials=n_trials,
    )
    logging.info("Number of finished trials: ", len(study.trials))

    logging.info("Best trial:")
    best_trial = study.best_trial

    logging.info(f"  Value: {best_trial.value}")

    logging.info("  Params: ")
    for key, value in best_trial.params.items():
        logging.info(f"    {key}: {value}")

    # Write report

    os.makedirs(config["tuning_logs_dir"], exist_ok=True)
    study.trials_dataframe().to_csv(
        os.path.join(config["tuning_logs_dir"], f"study_{model_class}_{timestamp}.csv")
    )

    return best_trial.params


def objective(
    trial: Trial,
    env_name: str,
    reward_class: type[BaseReward],
    config: dict[str, Any],
    hparam_space: dict[str, Any],
    max_training_iterations: int,
    n_eval_episodes: int,
    eval_freq: int,
    model_policy: type[ActorCriticPolicy],
    verbose: int,
) -> float:
    """Objective function that gets maximized.

    Args:
        trial (Trial): current trial
        env_name (str): env on which the trial is ran
        reward_class (type[BaseReward]): reward
        config (dict[str, Any]): configs to build the callback
        hparam_space (dict[str, Any]): hyperparameter space for sampling
        max_training_iterations (int): number of timesteps a single trial gets trained (if not pruned)
        n_eval_episodes (int): number of episodes to average to score the trial
        eval_freq (int): how frequently (in steps) a trial score is updated
        model_policy (type[ActorCriticPolicy]): which policy to use for PPO
        verbose (int): how much to log, 0 means as little as possible, 3 is the maximum.

    Raises:
        optuna.exceptions.TrialPruned: _description_

    Returns:
        float: _description_
    """
    hyperparameters = _sample_hyperparameters(trial, hparam_space)
    trial_eval_callback = TrialEvalCallback(
        env_name,
        trial,
        reward_class,
        config["obs_features"],
        config["selected_actions"],
        {"safe_max_rho": hyperparameters["safe_max_rho"]},
        config["chronics_filter"],
        config["seed"],
        n_eval_episodes,
        eval_freq,
        verbose,
    )

    train_topological_ppo(
        env_name,
        iterations=max_training_iterations,
        reward=reward_class,
        model_policy=model_policy,
        safe_max_rho=hyperparameters["safe_max_rho"],
        net_hyperparameters={
            hparam_name: hparam_value
            for hparam_name, hparam_value in hyperparameters.items()
            if hparam_name.startswith("net")
        },
        agent_hyperparameters={
            hparam_name: hparam_value
            for hparam_name, hparam_value in hyperparameters.items()
            if not hparam_name.startswith("net") and hparam_name != "safe_max_rho"
        },
        callbacks=[trial_eval_callback],
        prefix_folder=f"tuning_trial={trial.number}",
    )
    if trial_eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return trial_eval_callback.last_score


def _sample_hyperparameters(
    trial: Trial, hparam_space: dict[str, Any]
) -> dict[str, Any]:
    """Sample hyperparameters from dict.
    Simple language to specify hparams for optuna:
    - values => optuna draws among the values specified in the values list
        - n_draws => if n_draws is present together with values the output will be a list were n_draws will be made
        - repeat => if repeat is present together with values then the output will be a list were the draw is repeated 'repeat' times
    - low, high => a float value is sampled uniformly between low and high
        - log => if log=True means the value is sampled log uniformly

    Args:
        trial (Trial): optuna trial
        hparam_space (dict[str, Any]): hyperparameter space from which to sample

    Returns:
        dict[str, Any]: sampled hyperparameters
    """
    sampled_params = {}

    for param_name, param_spec in hparam_space.items():
        if isinstance(param_spec, dict):
            if "values" in param_spec:
                # Categorical parameter
                sampled_value = trial.suggest_categorical(
                    param_name, param_spec["values"]
                )
                if "repeat" in param_spec:
                    sampled_params[param_name] = [sampled_value] * param_spec["repeat"]
                elif "n_draws" in param_spec:
                    sampled_params[param_name] = [sampled_value] + [
                        trial.suggest_categorical(param_name, param_spec["values"])
                        for _ in range(param_spec["n_draws"] - 1)
                    ]
                else:
                    sampled_params[param_name] = sampled_value

            elif "low" in param_spec and "high" in param_spec:
                # Continuous parameter
                log_scale = param_spec.get("log", False)

                if log_scale:
                    sampled_params[param_name] = trial.suggest_loguniform(
                        param_name, param_spec["low"], param_spec["high"]
                    )
                else:
                    sampled_params[param_name] = trial.suggest_uniform(
                        param_name, param_spec["low"], param_spec["high"]
                    )

    # Handle batch_size constraint: batch_size = min(sampled_batch_size, sampled_n_steps)
    if "batch_size" in sampled_params and "n_steps" in sampled_params:
        sampled_params["batch_size"] = min(
            sampled_params["batch_size"], sampled_params["n_steps"]
        )

    return sampled_params
