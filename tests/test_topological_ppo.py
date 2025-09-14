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


def test_full_training():
    best_params = tune_topological_ppo(
        env_name=TEST_ENV,
        n_trials=5,
        model_class="PPO",
        n_sampler_startup_trials=3,
        min_timesteps_before_pruning=10,
        max_timesteps_per_trial=30,
        n_eval_episodes=5,
        eval_freq=15,
        verbose=1,
    )

    _, best_agent_name = train_topological_ppo(
        TEST_ENV,
        safe_max_rho=best_params[
            "safe_max_rho"
        ],  # thermal limit over which the agent acts to prevent blackouts
        net_hyperparameters={"net_arch": best_params["net_arch"]},  # MLP architecture
        agent_hyperparameters={
            param_name: param_value
            for param_name, param_value in best_params.items()
            if param_name not in {"safe_max_rho", "net_arch"}
        },  # all other PPO hyperparameters determined in the tuning step
        iterations=10,
    )

    evaluation = evaluate_topological_ppo(
        TEST_ENV,
        model=best_agent_name,
        n_eval_episodes=1,
    )

    print("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in evaluation:  # type: ignore [bad-unpacking]
        msg_tmp = "chronics at: {}".format(chron_name)
        msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
        msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)
