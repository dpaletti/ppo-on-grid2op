import logging
from typing import Any

import numpy as np
import optuna
from grid2op.Reward import BaseReward
from stable_baselines3.common.callbacks import BaseCallback

from ppo_on_grid2op.topological_ppo_evaluation import evaluate_topological_ppo


class TrialEvalCallback(BaseCallback):
    """Callback for evaluating and reporting trials during hyperparameter tuning.

    This callback allows the use of a custom evaluation function during
    hyperparameter optimization with Optuna. It evaluates the model at
    specified intervals and reports the results to Optuna for pruning
    decisions.

    Loosely based on: https://colab.research.google.com/github/araffin/tools-for-robotic-rl-icra2022/blob/main/notebooks/optuna_lab.ipynb#scrollTo=U5ijWTPzxSmd
    """

    def __init__(
        self,
        env_name: str,
        trial: optuna.Trial,
        reward_class: type[BaseReward],
        obs_features: list[str],
        selected_actions: list[str],
        gymenv_kwargs: dict[str, Any],
        chronics_filter: str,
        seed: int,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)

        self.env_name: str = env_name
        self.reward_class: type[BaseReward] = reward_class
        self.obs_features: list[str] = obs_features
        self.selected_actions: list[str] = selected_actions
        self.gymenv_kwargs: dict[str, Any] = gymenv_kwargs
        self.chronics_filter: str = chronics_filter
        self.seed: int = seed

        self.trial: optuna.Trial = trial
        self.n_eval_episodes: int = n_eval_episodes
        self.eval_freq: int = eval_freq
        self.eval_idx: int = 0

        self.is_pruned = False
        self.last_score = -np.inf
        self.verbose = verbose

    def _on_step(self) -> bool:
        """Perform evaluation and report to Optuna when scheduled.

        This method is (automatically) called after each environment step. It checks if
        evaluation should be performed based on the frequency setting,
        runs the custom evaluation function, reports results to Optuna,
        and handles pruning decisions.

        Returns:
            bool: True if training should continue, False if the trial
                should be pruned and training stopped.
        """
        continue_training: bool = True

        if self.n_calls % self.eval_freq == 0:
            # Run custom evaluation
            _, result = evaluate_topological_ppo(
                self.env_name,
                self.model,  # type: ignore[bad-argument-type]
                self.n_eval_episodes,
                self.reward_class,
                obs_features=self.obs_features,
                selected_actions=self.selected_actions,
                gymenv_kwargs=self.gymenv_kwargs,
                chronics_filter=self.chronics_filter,
                seed=self.seed,
                verbose=self.verbose > 0,
            )

            self.eval_idx += 1

            score = sum(
                [nb_time_step / max_ts for _, _, _, nb_time_step, max_ts in result]
            ) / len(result)
            self.last_score = score

            # Report to Optuna
            self.trial.report(score, self.eval_idx)

            # Prune trial if needed
            if self.trial.should_prune():
                if self.verbose > 0:
                    logging.info("Trial pruned by Optuna.")

                self.is_pruned = True
                continue_training = False

        return continue_training
