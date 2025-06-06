import os
import numpy as np
import torch
from src.chains.generation import MatrixGenerator, TensorGenerator
from src.chains.models import MarkovChainMatrix, MarkovChainTensor
from src.estimation.empirical import EmpiricalEstimator
from src.utils import chain_ten_to_mat

# TODO: What is the best way to generate different trials of the true MC as well?

class SimMatrixDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.generator = MatrixGenerator()
        self.estimator = EmpiricalEstimator()

        self.I = None
        self.trajectories = None
        self.mc_true = None
        self.mc_emp = None

    def generate_data(self):
        I = int(torch.prod(torch.tensor(self.cfg.dims)))

        mc = self.generator.lowrank(I, self.cfg.rank)
        trajectories = mc.simulate(
            num_steps=self.cfg.length,
            num_trajectories=self.cfg.trials,
            burn_in=self.cfg.burn_in,
        )
        estimates = self.estimator.estimate_matrix_batch(trajectories, I)
        P_emp_list, _ = zip(*estimates)
        P_true = mc.P

        mc_true = [MarkovChainMatrix(P_true) for _ in range(self.cfg.trials)]
        mc_emp = [MarkovChainMatrix(P) for P in P_emp_list]

        self.I = I
        self.trajectories = trajectories
        self.mc_true = mc_true
        self.mc_emp = mc_emp

        return trajectories, mc_true, mc_emp

    def save_data(self):
        assert all([getattr(self,attr) is not None for attr in {'I','trajectories','mc_true','mc_emp'}]), \
            "No data has been generated. Use `generate_data()` to create a dataset."
        assert self.cfg.data_path is not None, "No data path given."

        data_path = self.cfg.data_path
        P_true = self.mc_true[0].P
        P_emp_list = [self.mc_emp[t].P for t in range(self.cfg.trials)]
        trajectories = self.trajectories

        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        np.save(
            data_path,
            {
                "P_true": P_true.numpy(),
                "P_emp": [P.numpy() for P in P_emp_list],
                "trajectories": trajectories,
            },
        )
        
    def load_data(self):
        assert self.cfg.data_path is not None, "No data path given."
        data_path = self.cfg.data_path

        data = np.load(data_path, allow_pickle=True).item()
        P_true = torch.tensor(data["P_true"])
        P_emp_list = [torch.tensor(P) for P in data["P_emp"]]
        trajectories = data["trajectories"]

        if P_true.ndim>2:
            D = P_true.ndim // 2
            Is = torch.tensor(P_true.shape[:D])
            I = Is.prod().item()
            P_true = P_true.view(I, I)
            P_emp_list = [P.view(I, I) for P in P_emp_list]

            if len(trajectories[0][0])>1:
                trajectories = [chain_ten_to_mat(trajectory,Is) for trajectory in trajectories]

        self.cfg.trials = len(P_emp_list)
        self.cfg.dims = [P_true.shape[0]]
        self.cfg.length = len(trajectories[0])

        mc_true = [MarkovChainMatrix(P_true) for _ in range(self.cfg.trials)]
        mc_emp = [MarkovChainMatrix(P) for P in P_emp_list]

        self.I = P_true.shape[0]
        self.trajectories = trajectories
        self.mc_true = mc_true
        self.mc_emp = mc_emp

        return trajectories, mc_true, mc_emp
    
    def reset_data(self):
        self.I = None
        self.trajectories = None
        self.mc_true = None
        self.mc_emp = None

    def get_data(self):
        if all([getattr(self,attr) is not None for attr in {'I','trajectories','mc_true','mc_emp'}]):
            return self.trajectories, self.mc_true, self.mc_emp
        elif self.cfg.data_path is not None and os.path.exists(self.cfg.data_path):
            self.load_data()
            return self.trajectories, self.mc_true, self.mc_emp
        else:
            self.generate_data()
            return self.trajectories, self.mc_true, self.mc_emp


class SimTensorDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.generator = TensorGenerator()
        self.estimator = EmpiricalEstimator()

        self.I = None
        self.Is = None
        self.trajectories = None
        self.mc_true = None
        self.mc_emp = None

    def generate_data(self):
        dims = torch.tensor(self.cfg.dims)
        I = torch.prod(dims).item()

        mc = self.generator.lowrank(dims, self.cfg.rank)
        trajectories = mc.simulate(
            num_steps=self.cfg.length,
            num_trajectories=self.cfg.trials,
            burn_in=self.cfg.burn_in,
        )

        estimates = self.estimator.estimate_tensor_batch(trajectories, dims)
        P_emp_tensors, _ = zip(*estimates)
        P_true_tensor = mc.P

        P_true_mat = P_true_tensor.view(I, I)
        P_emp_mats = [P.view(I, I) for P in P_emp_tensors]

        P_true = P_true_mat.view(*dims, *dims)
        P_emp_list = [P.view(*dims, *dims) for P in P_emp_mats]

        mc_true = [MarkovChainTensor(P_true) for _ in range(self.cfg.trials)]
        mc_emp = [MarkovChainTensor(P) for P in P_emp_list]

        self.I = I
        self.Is = dims
        self.trajectories = trajectories
        self.mc_true = mc_true
        self.mc_emp = mc_emp

        return trajectories, mc_true, mc_emp

    def save_data(self):
        assert all([getattr(self,attr) is not None for attr in {'I','Is','trajectories','mc_true','mc_emp'}]), \
            "No data has been generated. Use `generate_data()` to create a dataset."
        assert self.cfg.data_path is not None, "No data path given."

        data_path = self.cfg.data_path

        P_true = self.mc_true[0].P
        P_emp_list = [self.mc_emp[t].P for t in range(self.cfg.trials)]
        # P_true_mat = self.mc_true[0].P.view(self.I, self.I)
        # P_emp_mats = [self.mc_emp[t].P.view(self.I, self.I) for t in range(self.cfg.trials)]
        trajectories = self.trajectories

        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        np.save(
            data_path,
            {
                # "P_true": P_true_mat.numpy(),
                # "P_emp": [P.numpy() for P in P_emp_mats],
                "P_true": P_true.numpy(),
                "P_emp": [P.numpy() for P in P_emp_list],
                "trajectories": trajectories,
            },
        )

    def load_data(self):
        assert self.cfg.data_path is not None, "No data path given."
        data_path = self.cfg.data_path

        data = np.load(data_path, allow_pickle=True).item()
        P_true = torch.tensor(data["P_true"])
        P_emp_list = [torch.tensor(P) for P in data["P_emp"]]
        trajectories = data["trajectories"]
        D = P_true.ndim // 2
        dims = torch.tensor(P_true.shape[:D])
        # P_true_mat = torch.tensor(data["P_true"])
        # P_true = P_true_mat.view(*self.cfg.dims, *self.cfg.dims)
        # P_emp_mats = [torch.tensor(P) for P in data["P_emp"]]
        # P_emp_list = [P.view(*self.cfg.dims, *self.cfg.dims) for P in P_emp_mats]
        # trajectories = data["trajectories"]

        self.cfg.dims = dims.tolist()
        self.cfg.trials = len(P_emp_list)
        self.cfg.length = len(trajectories[0])

        mc_true = [MarkovChainTensor(P_true) for _ in range(self.cfg.trials)]
        mc_emp = [MarkovChainTensor(P) for P in P_emp_list]

        self.I = torch.prod(dims).item()
        self.Is = dims
        self.trajectories = trajectories
        self.mc_true = mc_true
        self.mc_emp = mc_emp

        return trajectories, mc_true, mc_emp
    
    def reset_data(self):
        self.I = None
        self.Is = None
        self.trajectories = None
        self.mc_true = None
        self.mc_emp = None

    def get_data(self):
        if all([getattr(self,attr) is not None for attr in {'I','Is','trajectories','mc_true','mc_emp'}]):
            return self.trajectories, self.mc_true, self.mc_emp
        elif self.cfg.data_path is not None and os.path.exists(self.cfg.data_path):
            self.load_data()
            return self.trajectories, self.mc_true, self.mc_emp
        else:
            self.generate_data()
            return self.trajectories, self.mc_true, self.mc_emp