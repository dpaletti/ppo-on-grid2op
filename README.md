# PPO on Grid2Op
Experiments applying PPO algorithms to the Grid2Op RL environment.

## Reproducing the results
Clone the repository:
```bash
git clone https://github.com/dpaletti/ppo-on-grid2op.git
cd ppo-on-grid2op
```

Install all the required dependencies through the [UV](https://docs.astral.sh/uv/) python project manager:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Training scripts live in `src/ppo_on_grid2op/training_scripts`. To run any of those scripts it is sufficient to execute:
```bash
uv run python src/ppo_on_grid2op/training_scripts/train_baseline.py
```

For an even closer reproduction to how the results were actually computed in the folder `notebooks/reproducibility_notebooks` you find notebooks which can be directly ran on [Google Colab](https://colab.research.google.com/).


If one does not want to rerun all the jobs training and evaluation data are available under `presentation_data/`. Model zips are available upon request due to their size.

## Project map
All the code lives in `src/ppo_on_grid2op` so as to be in a packageable form in case it is needed to push it a python index for distribution.

The most important custom implementations are:
- `gnn_policy.py`: a stable-baselines actor-critic policy that works both for maskable and vanilla PPO
- `graph_gym_obs_space.py`: an observation adapter to be used to have a graph representation of the environment to the agent
- `maskable_sb3_agent.py`: a slight adaptation of the L2RPN baseline PPO agent to support masking through MaskablePPO from sb3-contrib
- `masked_env.py`: an environment to be used with MaskablePPO to provide action masks to the agent
- `trial_eval_callback.py`: callback to implement periodic pruning with optuna

Rudimentary tests are available in the `tests/` directory, these tests ensure that training jobs do not fail but do not test any other property or requirement.

`agent_evaluation.ipynb` and `plots.ipynb` inside the `notebooks/` directory provide all the necessary code to replicate plots and results from the data stored in this repository.