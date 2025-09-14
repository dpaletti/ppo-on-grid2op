import os

from grid2op.Agent.doNothing import DoNothingAgent
from grid2op.Agent.recoPowerlineAgent import RecoPowerlineAgent
from grid2op.Reward import LinesCapacityReward
from grid2op.Reward.baseReward import BaseReward
from grid2op.Runner import Runner

from ppo_on_grid2op.env_utils import make_seed_and_preload_grid2op_env
from ppo_on_grid2op.utils import read_config


def make_expert_system_baselines(
    env_name: str,
    reward_class: type[BaseReward] = LinesCapacityReward,
    verbose: bool = False,
    nb_episode: int = 100,
):
    """
    Create baseline performance from DoNothing and RecoAgent to compare other agents against.
    These baselines are useful to evaluate if the baseline agent we trained is meaningful with respect to strong expert systems whose rule get used during inference.
    In other words, we want to know what's the contribution of our agent on top of expert systems.
    """

    config = read_config()
    env = make_seed_and_preload_grid2op_env(
        env_name,
        reward_class,
        config["chronics_filter"],
        config["seed"],
        disable_cache=True,
    )

    do_nothing_agent = DoNothingAgent(env.action_space)
    reco_agent = RecoPowerlineAgent(env.action_space)

    for agent_name, agent in [
        ("do_nothing", do_nothing_agent),
        ("greedy_reconnection_agent", reco_agent),
    ]:
        runner_params = env.get_params_for_runner()
        runner_params["verbose"] = verbose
        runner = Runner(
            **runner_params,
            agentClass=None,
            agentInstance=agent,
        )  # type: ignore [missing-argument]

        # Run the agent on the scenarios
        os.makedirs(config["evaluation_logs_dir"], exist_ok=True)

        res = runner.run(
            path_save=os.path.join(config["evaluation_logs_dir"], agent_name),
            nb_episode=nb_episode,
            max_iter=-1,  # assess all the steps the agent is capable of doing
            pbar=verbose,
        )

        print(f"Evaluation summary for {agent_name}:")
        for _, chron_name, cum_reward, nb_time_step, max_ts in res:  # type: ignore [bad-unpacking]
            msg_tmp = "chronics at: {}".format(chron_name)
            msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
            msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
            print(msg_tmp)


if __name__ == "__main__":
    make_expert_system_baselines(
        "l2rpn_case14_sandbox_test",
        LinesCapacityReward,
        nb_episode=100,
        verbose=True,
    )
