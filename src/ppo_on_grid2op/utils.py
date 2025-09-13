import importlib.resources
import logging
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import torch.cuda

import ppo_on_grid2op.resources


def get_last_model_name() -> str:
    """Get last trained model out of the models directory.
    Models are expected to be one per directory and have a name such that
    {name}_env={env_name}_iterations={n_iterations}_%Y-%m-%d_%H:%M:%S

    Returns:
        str: name of the last trained model
    """
    folder_elements = list(Path(read_config()["models_dir"]).iterdir())
    training_dates: list[datetime] = []
    for element in folder_elements:
        if element.is_dir() and element.stem.split("_")[0] != "tuning":
            # Each directory contains a model, ignore tuning directories that contain several
            training_dates.append(
                datetime.strptime(
                    " ".join(element.stem.split("_")[-2:]), "%Y-%m-%d %H:%M:%S"
                )
            )

    last_trained_element = training_dates.index(max(training_dates))
    return folder_elements[last_trained_element].stem


def read_config() -> dict[str, Any]:
    """Read config from resources

    Returns:
        dict[str, Any]: config dict
    """
    return tomllib.loads(
        importlib.resources.read_text(ppo_on_grid2op.resources, "config.toml")
    )


def read_hyperparameters(model_class: str) -> dict[str, Any]:
    """Read hyperparameter TOML file.
    Expects to find a file named {model_class}_hparams.toml

    Args:
        model_class (str): model class to which the hyperparameter space belongs to

    Returns:
        dict[str, Any]: hyperparameter space ready to be parsed by optuna
    """
    return tomllib.loads(
        importlib.resources.read_text(
            ppo_on_grid2op.resources, f"{model_class}_hparams.toml"
        )
    )


def use_cuda() -> bool:
    """Whether cuda can be used or not.
    Checks in the config if cuda is selected, in that case check if hardware is detected by pytorch.

    Returns:
        bool: returns true if cuda is selected and hardware is detected, otherwise returns False
    """
    config = read_config()
    if config["use_cuda"] and not torch.cuda.is_available():
        logging.warning(
            "use_cuda set to true but torch cannot detect cuda hardware. CUDA not being used."
        )
        return False
    return True
