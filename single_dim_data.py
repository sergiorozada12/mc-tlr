import os
import torch
from omegaconf import OmegaConf
import argparse

from src.datasets.sim import SimDataset
from src.train import Trainer
from src.utils import chain_mat_to_ten, chain_ten_to_mat, ten_to_mat, mat_to_ten, qten_to_pten, qmat_to_pmat
from src.chains.generation import *

import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_base_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/data_single_dim")
    args = parser.parse_args()
    return args.config_base_path, args.data_dir

def main():
    config_base_path, data_dir = get_arguments()

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    cfg = OmegaConf.load(config_base_path)
    cfg.dataset.data_dir = f"{data_dir}/data_marg"

    dataset_marg = SimDataset(cfg.dataset)
    data_path = dataset_marg._get_full_path()

    if not os.path.exists(data_path):
        dataset_marg.generate_data(True)
        dataset_marg.save_data()
    else:
        dataset_marg.load_data(True)

    D = len(dataset_marg.Is)
    dataset_singles = [None for d in range(D)]
    dims_single = [None for d in range(D)]
    for d in range(D):
        cfg = OmegaConf.load(config_base_path)
        cfg.dataset.data_dir = f"{data_dir}/data_single{d}"
        cfg.dataset.length = 10
        cfg.dataset.dims = [dataset_marg.Is[d].item()]
        dims_single[d] = torch.tensor(cfg.dataset.dims)
        dataset_singles[d] = SimDataset(cfg.dataset)

    trajectories_true_single = [dataset_marg.trajectories_true[:,:,d].unsqueeze(2) for d in range(D)]
    trajectories_emp_single = [dataset_marg.trajectories_emp[:,:,d].unsqueeze(2) for d in range(D)]
    estimates = [dataset_singles[d].estimator.estimate_tensor_batch(trajectories_true_single[d],dims_single[d]) for d in range(D)]

    Q_singles = [dataset_marg.mc_true[0].Q.sum(tuple(np.delete(np.arange(2*D),[d,d+D]))) for d in range(D)]
    P_true_mats = [qmat_to_pmat(Q_singles[d]) for d in range(D)]
    P_emp_tensors = [list(zip(*estimates[d]))[0] for d in range(D)]
    P_emp_mats = [[ten_to_mat(P,dataset_marg.Is[d].item()) for P in P_emp_tensors[d]] for d in range(D)]

    for d in range(D):
        dataset_singles[d]._P_true_mat = P_true_mats[d]
        dataset_singles[d]._P_emp_list_mat = P_emp_mats[d]

        dataset_singles[d]._trajectories_emp_mat = torch.stack([torch.tensor(chain_ten_to_mat(traj,dims_single[d])) for traj in trajectories_emp_single[d]])
        dataset_singles[d]._trajectories_true_mat = torch.stack([torch.tensor(chain_ten_to_mat(traj,dims_single[d])) for traj in trajectories_true_single[d]])

    for d in range(D):
        dataset_singles[d].save_data()
        _ = dataset_singles[d].load_data(True)

if __name__=="__main__":
    main()
