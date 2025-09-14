from grid2op.Reward import EpisodeDurationReward, LinesCapacityReward
from sb3_contrib.ppo_mask.policies import MlpPolicy

from ppo_on_grid2op.topological_ppo_evaluation import evaluate_topological_ppo
from ppo_on_grid2op.topological_ppo_training import train_topological_ppo
from ppo_on_grid2op.topological_ppo_tuning import tune_topological_ppo

if __name__ == "__main__":
    env_name = "l2rpn_case14_sandbox"
    training_reward = LinesCapacityReward
    scoring_reward = EpisodeDurationReward
    tuning_timesteps = 1000  # computing poor
    model_policy = MlpPolicy

    best_params = tune_topological_ppo(
        env_name=env_name,
        n_trials=20,
        model_class="MaskablePPO",
        model_policy=model_policy,
        reward_class=training_reward,
        n_sampler_startup_trials=3,  # how many random trials to wait before starting the sampler
        min_timesteps_before_pruning=int(
            tuning_timesteps / 4
        ),  # how many timesteps to wait before evaluating to prune
        max_timesteps_per_trial=tuning_timesteps,  # how many timesteps for each trial
        hyperband_reduction_factor=3,  # how aggressive the pruner is (3 = balanced)
        n_eval_episodes=5,  # how many episodes average to evaluate model performance
        eval_freq=int(
            tuning_timesteps / 4
        ),  # frequency (in timesteps) of evaluations and subsequent pruning decisions
        enable_masking=True,
    )

    _, best_agent_name = train_topological_ppo(
        env_name,
        reward=training_reward,
        safe_max_rho=best_params[
            "safe_max_rho"
        ],  # thermal limit over which the agent acts to prevent blackouts
        net_hyperparameters={"net_arch": best_params["net_arch"]},  # MLP architecture
        agent_hyperparameters={
            param_name: param_value
            for param_name, param_value in best_params.items()
            if param_name not in {"safe_max_rho", "net_arch"}
        },  # all other PPO hyperparameters determined in the tuning step
        iterations=tuning_timesteps * 10,
        model_policy=model_policy,
        enable_masking=True,
    )

    evaluation = evaluate_topological_ppo(
        env_name,
        reward=scoring_reward,
        model=best_agent_name,
        n_eval_episodes=100,
        enable_masking=True,
    )

    print("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in evaluation:  # type: ignore [bad-unpacking]
        msg_tmp = "chronics at: {}".format(chron_name)
        msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
        msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)
