import argparse
import os
from omegaconf import OmegaConf
import torch
import wandb
import matplotlib
matplotlib.use("Agg")

from src.config import MainConfig
from src.train import Trainer
from src.datasets.sim import SimMatrixDataset, SimTensorDataset

def main(project, experiment_name, data_path):
    config = MainConfig()
    cfg = OmegaConf.structured(config)
    cfg.dataset.data_path = data_path
    torch.manual_seed(cfg.general.seed)
    torch.set_num_threads(1)

    dataset = SimMatrixDataset(cfg.dataset)
    trajectories, mc_true, mc_emp = dataset.get_data()
    Is = torch.tensor(cfg.dataset.dims)

    os.environ["WANDB_PROJECT"] = project

    trainer = Trainer(project=project, experiment_name=experiment_name, cfg=cfg)
    trainer.fit(trajectories, mc_true, mc_emp, Is)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",type=str,default="markov-chain-estimation")
    parser.add_argument("--experiment_name",type=str,default="default_experiment")
    parser.add_argument("--data_path",type=str,default="data/default_sim.npy")
    parser.add_argument("--sweep_config",type=str,default="default_sweep.yaml")
    args = parser.parse_args()
    
    project = args.project
    experiment_name = args.experiment_name
    sweep_config_path = args.sweep_config
    data_path = args.data_path

    assert os.path.exists(sweep_config_path), f"Sweep config file `{sweep_config_path}` does not exist."

    swp_config = OmegaConf.load(sweep_config_path)
    swp_cfg = OmegaConf.to_container(swp_config, resolve=True)
    swp_cfg.pop("program", None)
    swp_cfg['project'] = project

    def sweep_fn():
        main(project, experiment_name, data_path)

    sweep_id = wandb.sweep(swp_cfg, project=project)
    wandb.agent(sweep_id, function=sweep_fn)
