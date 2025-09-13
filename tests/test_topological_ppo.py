from ppo_on_grid2op.topological_ppo_evaluation import evaluate_topological_ppo
from ppo_on_grid2op.topological_ppo_training import train_topological_ppo
from ppo_on_grid2op.topological_ppo_tuning import tune_topological_ppo
from ppo_on_grid2op.utils import get_last_model_name

TEST_ENV = "l2rpn_case14_sandbox"


def test_train():
    train_topological_ppo(TEST_ENV, iterations=10)


def test_evaluate():
    evaluate_topological_ppo(
        env_name=TEST_ENV,
        model=get_last_model_name(),
        n_eval_episodes=10,
        n_parallel_evaluations=-1,
    )


def test_tuning():
    tune_topological_ppo(
        env_name=TEST_ENV,
        n_trials=10,
        model_class="PPO",
        n_sampler_startup_trials=1,
        min_timesteps_before_pruning=10,
        max_timesteps_per_trial=30,
        n_eval_episodes=1,
        eval_freq=15,
        verbose=1,
    )
