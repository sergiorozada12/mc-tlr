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
        I = int(torch.prod(torch.tensor(self.cfg.chain.dims)))

        mc = self.generator.lowrank(I, self.cfg.chain.rank)

        trajectories = mc.simulate(
            num_steps=self.cfg.chain.length,
            num_trajectories=self.cfg.general.trials,
            burn_in=self.cfg.chain.burn_in,
        )

        estimates = self.estimator.estimate_matrix_batch(trajectories, I)
        P_emp_list, _ = zip(*estimates)

        mc_true = [mc] * self.cfg.general.trials
        mc_emp = [MarkovChainMatrix(P) for P in P_emp_list]

        return trajectories, mc_true, mc_emp


class SimTensorDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.generator = TensorGenerator()
        self.estimator = EmpiricalEstimator()

    def get_data(self):
        dims = torch.tensor(self.cfg.chain.dims)

        mc = self.generator.lowrank(dims, self.cfg.chain.rank)

        trajectories = mc.simulate(
            num_steps=self.cfg.chain.length,
            num_trajectories=self.cfg.general.trials,
            burn_in=self.cfg.chain.burn_in,
        )

        estimates = self.estimator.estimate_tensor_batch(trajectories, dims)
        P_emp_list, _ = zip(*estimates)

        mc_true = [mc] * self.cfg.general.trials
        mc_emp = [MarkovChainTensor(P) for P in P_emp_list]

        return trajectories, mc_true, mc_emp
