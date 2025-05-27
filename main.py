import os
from omegaconf import OmegaConf
import torch
import wandb

from src.config import MainConfig
from src.train import Trainer
from src.datasets.sim import SimMatrixDataset, SimTensorDataset


def main():
    config = MainConfig()
    cfg = OmegaConf.structured(config)
    torch.manual_seed(cfg.general.seed)
    torch.set_num_threads(1)

    os.environ["WANDB_PROJECT"] = "markov-chain-estimation"

    # OVERWRITE CONFIG PARAMS HERE
    experiment_name = "traj_K5"
    cfg.method.method_name = "traj"
    cfg.method.fib.K = 5

    # LOAD DATA HERE
    dataset = SimTensorDataset(cfg)
    trajectories, mc_true, mc_emp = dataset.get_data()

    # LAUNCH TRAINER HERE
    trainer = Trainer(experiment_name=experiment_name, cfg=cfg, I=mc_true[0].I)
    trainer.fit(trajectories, mc_true, mc_emp)

    wandb.finish()


if __name__ == "__main__":
    main()
