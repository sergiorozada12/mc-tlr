import os
import torch
from omegaconf import OmegaConf
import argparse

from src.datasets.sim import SimDataset
from src.train import Trainer
from src.utils import chain_mat_to_ten, chain_ten_to_mat, ten_to_mat
from src.chains.generation import *

import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_base_path", type=str, required=True)
    parser.add_argument("--sample_range", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/data_vary_samples")
    return parser.parse_args()

def main():
    args = get_arguments()

    sample_range = eval(args.sample_range)
    assert type(sample_range)==list, "Expecting `sample_range` to be a list of integers."

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    config_base_path = args.config_base_path
    cfg = OmegaConf.load(config_base_path)
    cfg.dataset.length = int(sample_range[-1])

    # Generate or load full dataset
    cfg.dataset.data_dir = f"{data_dir}/samples{int(cfg.dataset.length)}"

    dataset_full = SimDataset(cfg.dataset)
    data_path = dataset_full._get_full_path()

    if not os.path.exists(data_path):
        dataset_full.generate_data(True)
        dataset_full.save_data()
    else:
        dataset_full.load_data(True)


    # Generate subsets of full dataset
    dims = torch.tensor(cfg.dataset.dims)
    I = torch.prod(dims).item()
    for i,N in enumerate(sample_range):
        cfg.dataset.data_dir = f"{data_dir}/samples{N}"
        cfg.dataset.length = 10
        dataset_subset = SimDataset(cfg.dataset)

        trajectories_emp_trunc = dataset_full.trajectories_emp[:,:sample_range[i]]
        trajectories_true_trunc = dataset_full.trajectories_true[:,:sample_range[i]]

        estimates = dataset_subset.estimator.estimate_tensor_batch(trajectories_true_trunc,dims)
        P_true_mat = ten_to_mat(dataset_full.mc_true[0].P, I)
        P_emp_tensors, _ = zip(*estimates)
        P_emp_mats = [ten_to_mat(P, I) for P in P_emp_tensors]

        dataset_subset._P_true_mat = P_true_mat
        dataset_subset._P_emp_list_mat = list(P_emp_mats)

        dataset_subset._trajectories_emp_mat = torch.stack([torch.tensor(chain_ten_to_mat(traj,dims)) for traj in trajectories_emp_trunc])
        dataset_subset._trajectories_true_mat = torch.stack([torch.tensor(chain_ten_to_mat(traj,dims)) for traj in trajectories_true_trunc])

        dataset_subset.save_data()
        _ = dataset_subset.load_data(True)

if __name__=="__main__":
    main()