import re
from datetime import datetime

import grid2op
from grid2op.Chronics import MultifolderWithCache
from grid2op.Reward import EpisodeDurationReward
from l2rpn_baselines.PPO_SB3 import train
from l2rpn_baselines.PPO_SB3.utils import SB3Agent
from lightsim2grid import LightSimBackend  # type: ignore[possibly-unbound-import]

from ppo_on_grid2op.env_with_heuristics import (
    GymEnvWithRecoWithDNWithShuffle,
)
from ppo_on_grid2op.utils import read_config, use_cuda


def train_baseline(env_name: str, iterations: int, verbose: bool = False) -> SB3Agent:
    config = read_config()
    env = grid2op.make(
        env_name,
        backend=LightSimBackend(),
        chronics_class=MultifolderWithCache,
        reward_class=EpisodeDurationReward,
    )

    env.chronics_handler.real_data.set_filter(lambda x: re.match(".*0$", x) is not None)
    env.chronics_handler.real_data.reset()

    timestamp = "".join(str(datetime.now()).split(".")[0:-1]).replace(" ", "_")

    # TODO: move to optuna HPT
    PPO_HP = {
        "net_arch": [100, 100, 100],
        "gamma": 0.99,
        "n_steps": 16,
        "batch_size": 16,
        "learning_rate": 3e-6,
    }
    gymenv_kwargs = {"safe_max_rho": 0.99}

    return train(
        env,
        act_attr_to_keep=config["selected_actions"],
        obs_attr_to_keep=config["obs_features"],
        save_path=config["models_dir"],
        logs_dir=config["tensorboard_logs_dir"],
        iterations=iterations,
        name=f"PPO_env={env_name}_iterations={iterations}_{timestamp}_",
        verbose=1,
        gymenv_class=GymEnvWithRecoWithDNWithShuffle,
        device="cuda" if use_cuda() else "cpu",
        normalize_act=True,
        normalize_obs=True,
        save_every_xxx_steps=min(max(iterations // 10, 1), 1_000_000),
        seed=config["seed"],
        gymenv_kwargs=gymenv_kwargs,
        **PPO_HP,
    )


# TODO: develop evaluation section through the evaluate() method
