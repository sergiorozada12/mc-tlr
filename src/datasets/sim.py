import os
import numpy as np
import torch

from src.chains.generation import MatrixGenerator, TensorGenerator
from src.chains.models import MarkovChainMatrix, MarkovChainTensor
from src.estimation.empirical import EmpiricalEstimator
from src.utils import (
    chain_mat_to_ten,
    mat_to_ten,
    ten_to_mat,
)


class SimDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.generator = (
            MatrixGenerator() if cfg.generation_type == "matrix" else TensorGenerator()
        )
        self.estimator = EmpiricalEstimator()

        self.I = None
        self.Is = None
        self.mc_true = None
        self.mc_emp = None
        self.trajectories_true = None
        self.trajectories_emp = None
        self._P_true_mat = None
        self._P_emp_list_mat = None

    def _get_filename(self):
        return f"sim_{self.cfg.generation_type}_{self.cfg.generation_method}.npy"

    def _get_full_path(self):
        return os.path.join(self.cfg.data_dir, self._get_filename())

    def generate_data(self, use_tensor: bool):
        dims = torch.tensor(self.cfg.dims)
        I = torch.prod(dims).item()
        self.I = I
        self.Is = dims

        gen_method = getattr(self.generator, self.cfg.generation_method)
        if self.cfg.generation_type == "matrix":
            mc = gen_method(I, self.cfg.rank)
            trajectories = mc.simulate(
                num_steps=self.cfg.length,
                num_trajectories=self.cfg.trials,
                burn_in=self.cfg.burn_in,
            )
            estimates = self.estimator.estimate_matrix_batch(trajectories, I)
            P_true_mat = mc.P
            P_emp_mats, _ = zip(*estimates)
        else:
            mc = gen_method(dims, self.cfg.rank)
            trajectories = mc.simulate(
                num_steps=self.cfg.length,
                num_trajectories=self.cfg.trials,
                burn_in=self.cfg.burn_in,
            )
            estimates = self.estimator.estimate_tensor_batch(trajectories, dims)
            P_true_mat = ten_to_mat(mc.P, I)
            P_emp_tensors, _ = zip(*estimates)
            P_emp_mats = [ten_to_mat(P, I) for P in P_emp_tensors]

        self._P_true_mat = P_true_mat
        self._P_emp_list_mat = list(P_emp_mats)

        mc_true = [MarkovChainMatrix(P_true_mat) for _ in range(self.cfg.trials)]
        mc_emp = [MarkovChainMatrix(P) for P in P_emp_mats]

        trajectories_true = [
            mc.simulate(
                num_steps=self.cfg.length,
                num_trajectories=1,
                burn_in=self.cfg.burn_in,
            )[0]
            for mc in mc_true
        ]
        trajectories_emp = [
            mc.simulate(
                num_steps=self.cfg.length,
                num_trajectories=1,
                burn_in=self.cfg.burn_in,
            )[0]
            for mc in mc_emp
        ]

        if use_tensor:
            trajectories_true = [chain_mat_to_ten(traj, dims) for traj in trajectories_true]
            trajectories_emp = [chain_mat_to_ten(traj, dims) for traj in trajectories_emp]
            P_true_tensor = mat_to_ten(P_true_mat, dims)
            P_emp_tensors = [mat_to_ten(P, dims) for P in P_emp_mats]
            mc_true = [MarkovChainTensor(P_true_tensor) for _ in range(self.cfg.trials)]
            mc_emp = [MarkovChainTensor(P) for P in P_emp_tensors]
            trajectories_true = torch.stack([
                torch.stack([torch.tensor(x, dtype=torch.long) for x in traj])
                for traj in trajectories_true
            ])
            trajectories_emp = torch.stack([
                torch.stack([torch.tensor(x, dtype=torch.long) for x in traj])
                for traj in trajectories_emp
            ])
        else:
            trajectories_true = torch.tensor(trajectories_true, dtype=torch.long)
            trajectories_emp = torch.tensor(trajectories_emp, dtype=torch.long)

        self.mc_true = mc_true
        self.mc_emp = mc_emp
        self.trajectories_true = trajectories_true
        self.trajectories_emp = trajectories_emp

        self.save_data()
        return trajectories_true, trajectories_emp, mc_true, mc_emp

    def save_data(self):
        assert self._P_true_mat is not None and self._P_emp_list_mat is not None

        data_path = self._get_full_path()
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        np.save(
            data_path,
            {
                "P_true": self._P_true_mat.numpy(),
                "P_emp": [P.numpy() for P in self._P_emp_list_mat],
                "trajectories_true": self.trajectories_true.numpy(),
                "trajectories_emp": self.trajectories_emp.numpy(),
                "dims": self.cfg.dims,
            },
        )

    def load_data(self, use_tensor: bool):
        data_path = self._get_full_path()
        data = np.load(data_path, allow_pickle=True).item()

        P_true = torch.tensor(data["P_true"])
        P_emp_list = [torch.tensor(P) for P in data["P_emp"]]
        trajectories_true = data["trajectories_true"]
        trajectories_emp = data["trajectories_emp"]
        dims = torch.tensor(data.get("dims", [P_true.shape[0]]))
        I = torch.prod(dims).item()

        self.I = I
        self.Is = dims
        self.cfg.dims = dims.tolist()
        self.cfg.trials = len(P_emp_list)
        self.cfg.length = trajectories_true.shape[1]

        mc_true = [MarkovChainMatrix(P_true) for _ in range(self.cfg.trials)]
        mc_emp = [MarkovChainMatrix(P) for P in P_emp_list]

        if use_tensor:
            trajectories_true = torch.tensor(trajectories_true, dtype=torch.long)
            trajectories_emp = torch.tensor(trajectories_emp, dtype=torch.long)
            P_true_tensor = mat_to_ten(P_true, dims)
            P_emp_tensors = [mat_to_ten(P, dims) for P in P_emp_list]
            mc_true = [MarkovChainTensor(P_true_tensor) for _ in range(self.cfg.trials)]
            mc_emp = [MarkovChainTensor(P) for P in P_emp_tensors]
        else:
            trajectories_true = torch.tensor(trajectories_true, dtype=torch.long)
            trajectories_emp = torch.tensor(trajectories_emp, dtype=torch.long)

        self.mc_true = mc_true
        self.mc_emp = mc_emp
        self.trajectories_true = trajectories_true
        self.trajectories_emp = trajectories_emp

        return trajectories_true, trajectories_emp, mc_true, mc_emp

    def get_data(self, use_tensor: bool = False):
        if os.path.exists(self._get_full_path()):
            return self.load_data(use_tensor)
        else:
            return self.generate_data(use_tensor)
