from grid2op.Reward import EpisodeDurationReward
from stable_baselines3.ppo import MlpPolicy

from ppo_on_grid2op.topological_ppo_evaluation import evaluate_topological_ppo
from ppo_on_grid2op.topological_ppo_training import train_topological_ppo
from ppo_on_grid2op.topological_ppo_tuning import tune_topological_ppo

if __name__ == "__main__":
    env_name = "l2rpn_case14_sandbox"
    reward = EpisodeDurationReward  # returns number of steps seen before blackout
    tuning_timesteps = 1000  # computing poor
    model_policy = MlpPolicy  # baseline policy

    print("Starting hyperparameter tuning.")
    best_params = tune_topological_ppo(
        env_name=env_name,
        n_trials=100,
        model_class="PPO",
        model_policy=model_policy,
        reward_class=reward,
        n_sampler_startup_trials=3,  # how many random trials to wait before starting the sampler
        min_timesteps_before_pruning=int(
            tuning_timesteps / 4
        ),  # how many timesteps to wait before evaluating to prune
        max_timesteps_per_trial=tuning_timesteps,  # how many timesteps for each trial
        hyperband_reduction_factor=3,  # how aggressive the pruner is (3 = balanced)
        n_eval_episodes=5,  # how many episodes average to evaluate model performance
        eval_freq=int(
            tuning_timesteps / 3
        ),  # frequency (in timesteps) of evaluations and subsequent pruning decisions
        verbose=0,
    )

    print("Starting training of the best model.")
    # Here one could also take over from the hyperparameter tuning
    # For clarity and simplicity I start from scratch
    _, best_agent_name = train_topological_ppo(
        env_name=env_name,
        iterations=tuning_timesteps
        * 10,  # should be higher but my laptop (and colab) is crying
        reward=reward,
        model_policy=model_policy,
        safe_max_rho=best_params[
            "safe_max_rho"
        ],  # thermal limit over which the agent acts to prevent blackouts
        net_hyperparameters={"net_arch": best_params["net_arch"]},  # MLP architecture
        agent_hyperparameters={
            param_name: param_value
            for param_name, param_value in best_params.items()
            if param_name not in {"safe_max_rho", "net_arch"}
        },  # all other PPO hyperparameters determined in the tuning step
        verbose=0,
    )

    print("Starting evaluation of the best model trained.")
    evaluation = evaluate_topological_ppo(
        env_name=env_name,
        model=best_agent_name,
        n_eval_episodes=100,
        reward=reward,
        verbose=True,
    )

    print("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in evaluation:  # type: ignore [bad-unpacking]
        msg_tmp = "chronics at: {}".format(chron_name)
        msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
        msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)
