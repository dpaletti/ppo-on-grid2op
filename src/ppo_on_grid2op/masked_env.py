import numpy as np

from ppo_on_grid2op.env_with_heuristics import GymEnvWithRecoWithDNWithShuffle


class MaskedGymEnvWithRecoWithDNWithShuffle(GymEnvWithRecoWithDNWithShuffle):
    """This environment implements the action_mask method so that maskable models (e.g. MaskablePPO) retrieves the mask at learning time."""

    def __init__(
        self,
        env_init,
        *args,
        reward_cumul="init",
        safe_max_rho=0.9,
        apply_validity_mask=True,
        disable_shuffle=False,
        **kwargs,
    ):
        super().__init__(
            env_init,
            *args,
            reward_cumul=reward_cumul,
            safe_max_rho=safe_max_rho,
            disable_shuffle=disable_shuffle,
            **kwargs,
        )
        self.apply_validity_mask = apply_validity_mask

    def action_masks(self):
        if self.apply_validity_mask:
            return self._check_action_legal_and_not_ambiguous()
        else:
            # Return all actions as valid if masking is disabled
            return np.ones(self.action_space.n, dtype=bool)

    def _check_action_legal_and_not_ambiguous(self):
        mask = np.ones(self.action_space.n, dtype=bool)

        for action_idx in range(self.action_space.n):
            grid2op_action = self.action_space.converter.convert_act(action_idx)

            if not self.action_space.converter.legal_action(
                grid2op_action, self.init_env
            )[0]:
                mask[action_idx] = False

        return mask
