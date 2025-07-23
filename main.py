import argparse
import os
import torch
import wandb
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")

from src.datasets.sim import SimDataset
from src.datasets.taxi import TaxiDataset
from src.train import Trainer


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default="markov-chain-estimation")
    parser.add_argument("--wandb_experiment_name", type=str, default="default_experiment")
    parser.add_argument("--config_dataset", type=str, required=True)
    parser.add_argument("--config_base_path", type=str, required=True)
    parser.add_argument("--config_sweep_path", type=str, required=True)
    return parser.parse_args()


def run_training(project, experiment_name, config_dataset, config_base_path):
    wandb.init(project=project)

    # GET CONFIG
    base_cfg = OmegaConf.load(config_base_path)
    sweep_cfg = OmegaConf.create(wandb.config.as_dict())
    cfg = OmegaConf.merge(base_cfg, sweep_cfg)

    # CONFIGURE WANDB
    method_name = sweep_cfg.method.method_name
    param_cfg = sweep_cfg.method[method_name]
    param_str = "-".join(f"{k}{v}" for k, v in param_cfg.items())
    run_name = f"{method_name}_{param_str}"

    wandb.run.name = run_name
    wandb.run.save()

    os.environ["WANDB_PROJECT"] = project
    torch.manual_seed(cfg.general.seed)
    torch.set_num_threads(1)

    # GET DATA
    if config_dataset == "sim":
        dataset = SimDataset(cfg.dataset)
    elif config_dataset == "taxi":
        dataset = TaxiDataset(cfg.dataset)
    else:
        raise "Dataset not supported."

    # RUN EXPERIMENT
    trainer = Trainer(
        project=project,
        experiment_name=experiment_name,
        cfg=cfg,
        log_results=True,
        save_results=False,
    )
    trainer.fit(dataset)

    wandb.finish()


def main():
    args = get_arguments()
    project = args.wandb_project
    experiment_name = args.wandb_experiment_name
    config_dataset = args.config_dataset
    config_base_path = args.config_base_path
    config_sweep_path = args.config_sweep_path

    sweep_yaml = OmegaConf.load(config_sweep_path)
    sweep_dict = OmegaConf.to_container(sweep_yaml, resolve=True)
    sweep_dict.pop("program", None)
    sweep_dict["project"] = project

    sweep_id = wandb.sweep(sweep_dict, project=project)
    wandb.agent(sweep_id, function=lambda: run_training(project, experiment_name, config_dataset, config_base_path))


if __name__ == "__main__":
    main()