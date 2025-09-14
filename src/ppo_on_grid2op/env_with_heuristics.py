from typing import Any

from grid2op.Chronics.multiFolder import Multifolder
from gymnasium.core import ObservationWrapper
from l2rpn_baselines.utils import GymEnvWithRecoWithDN


class GymEnvWithRecoWithDNWithShuffle(GymEnvWithRecoWithDN):
    """
    Environment with built-in heuristics:
    - reconnect all powerlines whenever possible
    - do nothing until thermal limit (rho) exceeds safe_max_rho

    Slight adaptation of https://github.com/gaetanserre/L2RPN-2022_PPO-Baseline/blob/main/src/GymEnvWithRecoWithDNWithShuffle.py
    """

    def __init__(
        self,
        env_init,
        *args,
        reward_cumul="init",
        safe_max_rho=0.9,
        disable_shuffle: bool = False,
        **kwargs,
    ):
        super().__init__(
            env_init,
            *args,
            reward_cumul=reward_cumul,
            safe_max_rho=safe_max_rho,
            **kwargs,
        )
        self.nb_reset = 0
        self.disable_shuffle = disable_shuffle

    def reset(
        self,
        seed: int | None = None,
        return_info: bool = False,
        options: dict[str, Any] | None = None,
    ) -> ObservationWrapper:
        """Episode reset, shuffle the chronics every time all the chronics have been played once.

        Args:
            seed (int | None, optional): Randomness reproducibility parameter. Defaults to None.
            return_info (bool, optional): wheter to return info. Defaults to False.
            options (dict[str, Any] | None, optional): extra options to pass to the env downstream. Defaults to None.

        Returns:
            ObservationWrapper: first observation of the new episode
        """
        if self.disable_shuffle:
            return super().reset(seed=seed, return_info=return_info, options=options)  # type: ignore[bad-return]

        self.nb_reset += 1
        if isinstance(self.init_env.chronics_handler.real_data, Multifolder):  # type: ignore[missing-attribute]
            nb_chron = len(self.init_env.chronics_handler.real_data._order)  # type: ignore[missing-attribute, bad-argument-type]
            if self.nb_reset % nb_chron == 0:
                self.init_env.chronics_handler.reset()  # type: ignore[missing-attribute]
        return super().reset(seed=seed, return_info=return_info, options=options)  # type: ignore[bad-return]
