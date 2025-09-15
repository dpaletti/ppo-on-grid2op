from grid2op.Reward import EpisodeDurationReward

from ppo_on_grid2op.env_utils import make_discrete_action_gym_env
from ppo_on_grid2op.gnn_policy import ActorCriticGNNPolicy, MaskableActorCriticGNNPolicy
from ppo_on_grid2op.topological_ppo_evaluation import evaluate_topological_ppo
from ppo_on_grid2op.topological_ppo_training import train_topological_ppo
from ppo_on_grid2op.topological_ppo_tuning import (
    split_net_and_agent_params,
    tune_topological_ppo,
)
from ppo_on_grid2op.utils import get_last_model_name, read_config

TEST_ENV = "l2rpn_case14_sandbox"
config = read_config()


def test_make_graph_env():
    gym_env, env = make_discrete_action_gym_env(
        TEST_ENV,
        "train",
        [],
        ["set_bus"],
        EpisodeDurationReward,
        {"safe_max_rho": 0.99},
        chronics_filter=config["chronics_filter"],
        seed=config["seed"],
        enable_graph=True,
    )


def test_train():
    train_topological_ppo(
        TEST_ENV,
        iterations=10,
        model_name="GraphPPO",
        enable_graph=True,
        model_policy=ActorCriticGNNPolicy,
    )


def test_evaluate():
    evaluate_topological_ppo(
        env_name=TEST_ENV,
        model=get_last_model_name("GraphPPO"),
        n_eval_episodes=10,
        n_parallel_evaluations=1,  # -1 deadlocks (probaby due to GNN)
        enable_graph=True,
        verbose=True,
    )


def test_tuning():
    tune_topological_ppo(
        env_name=TEST_ENV,
        n_trials=10,
        model_class="GraphPPO",
        n_sampler_startup_trials=1,
        min_timesteps_before_pruning=10,
        max_timesteps_per_trial=30,
        n_eval_episodes=1,
        eval_freq=15,
        verbose=1,
        enable_masking=False,
        model_policy=ActorCriticGNNPolicy,
        enable_graph=True,
    )


def test_full_training():
    best_params = tune_topological_ppo(
        env_name=TEST_ENV,
        n_trials=2,
        model_class="GraphPPO",
        n_sampler_startup_trials=3,
        min_timesteps_before_pruning=10,
        max_timesteps_per_trial=30,
        n_eval_episodes=5,
        eval_freq=15,
        verbose=1,
        enable_masking=False,
        model_policy=ActorCriticGNNPolicy,
        enable_graph=True,
    )
    net_hparams, agent_hparams = split_net_and_agent_params(best_params)

    _, best_agent_name = train_topological_ppo(
        TEST_ENV,
        safe_max_rho=best_params[
            "safe_max_rho"
        ],  # thermal limit over which the agent acts to prevent blackouts
        net_hyperparameters=net_hparams,
        agent_hyperparameters=agent_hparams,
        iterations=10,
        model_policy=ActorCriticGNNPolicy,
        enable_graph=True,
    )

    evaluation = evaluate_topological_ppo(
        TEST_ENV,
        model=best_agent_name,
        n_eval_episodes=1,
        enable_graph=True,
    )

    print("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in evaluation:  # type: ignore [bad-unpacking]
        msg_tmp = "chronics at: {}".format(chron_name)
        msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
        msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)


def test_maskable_full_training():
    best_params = tune_topological_ppo(
        env_name=TEST_ENV,
        n_trials=3,
        model_class="MaskableGraphPPO",
        n_sampler_startup_trials=3,
        min_timesteps_before_pruning=10,
        max_timesteps_per_trial=30,
        n_eval_episodes=5,
        eval_freq=15,
        verbose=1,
        model_policy=MaskableActorCriticGNNPolicy,
        enable_graph=True,
        enable_masking=True,
    )
    net_hparams, agent_hparams = split_net_and_agent_params(best_params)

    _, best_agent_name = train_topological_ppo(
        TEST_ENV,
        safe_max_rho=best_params[
            "safe_max_rho"
        ],  # thermal limit over which the agent acts to prevent blackouts
        net_hyperparameters=net_hparams,
        agent_hyperparameters=agent_hparams,
        iterations=10,
        model_policy=MaskableActorCriticGNNPolicy,
        enable_graph=True,
        enable_masking=True,
    )

    evaluation = evaluate_topological_ppo(
        TEST_ENV,
        model=best_agent_name,
        n_eval_episodes=1,
        enable_masking=True,
        enable_graph=True,
    )

    print("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in evaluation:  # type: ignore [bad-unpacking]
        msg_tmp = "chronics at: {}".format(chron_name)
        msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
        msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)
