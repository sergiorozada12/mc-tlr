import os
from omegaconf import OmegaConf
import torch
import wandb

from src.config import MainConfig
from src.chains.generation import TensorGenerator
from src.train import Trainer
from src.chains.models import MarkovChainTensor
from src.utils import ten2mat
from src.estimation.empirical import EmpiricalEstimator  # or wherever it's defined


def main():
    config = MainConfig()
    cfg = OmegaConf.structured(config)
    torch.manual_seed(cfg.general.seed)
    torch.set_num_threads(1)

    # Set up wandb project
    os.environ["WANDB_PROJECT"] = "markov-chain-estimation"

    # Generate data
    generator = TensorGenerator()
    dims = torch.tensor(cfg.chain.dims)
    mc = generator.lowrank(dims, cfg.chain.rank)
    trajectories = mc.simulate(
        num_steps=cfg.chain.length,
        num_trajectories=cfg.general.trials,
        burn_in=cfg.chain.burn_in,
    )

    # Estimate empirical models from trajectories
    estimator = EmpiricalEstimator()
    estimates = estimator.estimate_tensor_batch(trajectories, dims)
    P_emp_list, Q_emp_list = zip(*estimates)
    mc_emp = [MarkovChainTensor(P) for P in P_emp_list]

    # Ground-truth matrix version of the model (for comparison)
    mc_true = [mc.to_matrix()] * cfg.general.trials

    # Train
    trainer = Trainer(method_name=cfg.method.method_name, cfg=cfg, I=mc_true[0].I)
    trainer.fit(trajectories, mc_true, mc_emp)

    wandb.finish()


if __name__ == "__main__":
    main()