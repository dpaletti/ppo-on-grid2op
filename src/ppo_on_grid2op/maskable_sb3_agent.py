from l2rpn_baselines.PPO_SB3.utils import SB3Agent
from sb3_contrib import MaskablePPO


class MaskableSB3Agent(SB3Agent):
    def __init__(
        self,
        g2op_action_space,
        gym_act_space,
        gym_obs_space,
        nn_type=MaskablePPO,  # this is the only change to the init
        nn_path=None,
        nn_kwargs=None,
        custom_load_dict=None,
        gymenv=None,
        iter_num=None,
    ):
        super().__init__(
            g2op_action_space,
            gym_act_space,
            gym_obs_space,
            nn_type,
            nn_path,
            nn_kwargs,
            custom_load_dict,
            gymenv,
            iter_num,
        )
        self.gym_env = gymenv  # For grid2op runner compatibility

    def get_act(self, gym_obs, reward, done):
        """Retrieve the gym action and the action mask from the gym observation and the reward.

        Parameters
        ----------
        gym_obs : gym observation
            The gym observation
        reward : ``float``
            the current reward
        done : ``bool``
            whether the episode is over or not.

        Returns
        -------
        gym action
            The gym action, that is processed in the :func:`GymAgent.act`
            to be used with grid2op
        """
        action_masks = self.gym_env.action_masks()
        action, _ = self.nn_model.predict(
            gym_obs, deterministic=True, action_masks=action_masks
        )
        return action

    def build(self):
        """Create the underlying NN model from scratch.

        In the case of a PPO agent, this is equivalent to perform the:

        .. code-block:: python

            PPO(**nn_kwargs)
        """
        self.nn_model = MaskablePPO(**self._nn_kwargs)  # type: ignore
