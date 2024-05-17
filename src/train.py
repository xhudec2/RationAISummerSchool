# Copyright (c) The RationAI team.

import random

import hydra
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf


OmegaConf.register_new_resolver(
    "random_seed", lambda: random.randint(0, 2**31), use_cache=True
)


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def train(config: DictConfig) -> None:
    seed_everything(config.seed)

    data = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model)

    trainer = hydra.utils.instantiate(config.trainer)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter
