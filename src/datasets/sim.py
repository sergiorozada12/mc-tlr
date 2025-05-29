import os
import numpy as np
import torch
from src.chains.generation import MatrixGenerator, TensorGenerator
from src.chains.models import MarkovChainMatrix, MarkovChainTensor
from src.estimation.empirical import EmpiricalEstimator

class SimMatrixDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.generator = MatrixGenerator()
        self.estimator = EmpiricalEstimator()

    def get_data(self):
        I = int(torch.prod(torch.tensor(self.cfg.dims)))
        data_path = self.cfg.data_path

        if data_path is not None and os.path.exists(data_path):
            data = np.load(data_path, allow_pickle=True).item()
            P_true = torch.tensor(data["P_true"])
            P_emp_list = [torch.tensor(P) for P in data["P_emp"]]
            trajectories = data["trajectories"]
        else:
            mc = self.generator.lowrank(I, self.cfg.rank)
            trajectories = mc.simulate(
                num_steps=self.cfg.length,
                num_trajectories=self.cfg.trials,
                burn_in=self.cfg.burn_in,
            )
            estimates = self.estimator.estimate_matrix_batch(trajectories, I)
            P_emp_list, _ = zip(*estimates)
            P_true = mc.P

            if data_path is not None:
                os.makedirs(os.path.dirname(data_path), exist_ok=True)
                np.save(data_path, {
                    "P_true": P_true.numpy(),
                    "P_emp": [P.numpy() for P in P_emp_list],
                    "trajectories": trajectories,
                })

        mc_true = [MarkovChainMatrix(P_true) for _ in range(self.cfg.trials)]
        mc_emp = [MarkovChainMatrix(P) for P in P_emp_list]

        return trajectories, mc_true, mc_emp


class SimTensorDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.generator = TensorGenerator()
        self.estimator = EmpiricalEstimator()

    def get_data(self):
        dims = torch.tensor(self.cfg.dims)
        data_path = self.cfg.data_path

        if data_path is not None and os.path.exists(data_path):
            data = np.load(data_path, allow_pickle=True).item()
            P_true = torch.tensor(data["P_true"])
            P_emp_list = [torch.tensor(P) for P in data["P_emp"]]
            trajectories = data["trajectories"]
        else:
            mc = self.generator.lowrank(dims, self.cfg.rank)
            trajectories = mc.simulate(
                num_steps=self.cfg.length,
                num_trajectories=self.cfg.trials,
                burn_in=self.cfg.burn_in,
            )
            estimates = self.estimator.estimate_tensor_batch(trajectories, dims)
            P_emp_list, _ = zip(*estimates)
            P_true = mc.P

            if data_path is not None:
                os.makedirs(os.path.dirname(data_path), exist_ok=True)
                np.save(data_path, {
                    "P_true": P_true.numpy(),
                    "P_emp": [P.numpy() for P in P_emp_list],
                    "trajectories": trajectories,
                })

        mc_true = [MarkovChainTensor(P_true) for _ in range(self.cfg.trials)]
        mc_emp = [MarkovChainTensor(P) for P in P_emp_list]

        return trajectories, mc_true, mc_emp