import torch
import numpy as np
from abc import ABC, abstractmethod

from src.utils import mat2ten, ten2mat, normalize_rows, normalize_tensor_rows


class BaseMarkovChain(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self):
        pass

    def simulate(self, num_steps: int, num_trajectories: int = 1, burn_in: int = 0):
        trajectories = []
        for _ in range(num_trajectories):
            self.reset()
            trajectory = [self.current_state.clone() if isinstance(self.current_state, torch.Tensor) else self.current_state]
            for _ in range(1, num_steps + burn_in):
                trajectory.append(self.step())
            trajectories.append(trajectory[burn_in:])
        return trajectories


class MarkovChainMatrix(BaseMarkovChain):
    def __init__(self, P: torch.Tensor):
        assert P.ndim == 2 and P.shape[0] == P.shape[1], "P must be square matrix"
        assert torch.allclose(P.sum(dim=1), torch.ones(P.shape[0])), "Rows must sum to 1"
        assert (P >= 0).all() and (P <= 1).all(), "Invalid probabilities in P"

        self.P = normalize_rows(P)
        self.I = P.shape[0]

        evals, evecs = torch.linalg.eig(self.P.T)
        idx_pi = torch.where(torch.abs(evals - 1) <= 1e-5)[0][0]
        self.R = torch.abs(torch.real(evecs[:, idx_pi]))
        self.R = self.R / self.R.sum()

        self.Q = torch.diag(self.R) @ self.P
        self.current_state = torch.multinomial(self.R, 1).item()

    def reset(self):
        self.current_state = torch.multinomial(self.R, 1).item()

    def step(self):
        transition_prob = self.P[self.current_state]
        self.current_state = torch.multinomial(transition_prob, 1).item()
        return self.current_state


class MarkovChainTensor(BaseMarkovChain):
    def __init__(self, P: torch.Tensor):
        assert P.ndim % 2 == 0, "P must have even number of dimensions"
        D = P.ndim // 2
        assert P.shape[:D] == P.shape[D:], "Shape mismatch between source and target dimensions"
        assert (P >= 0).all() and (P <= 1).all(), "Invalid probabilities in P"

        self.D = D
        self.Is = torch.tensor(P.shape[:D])
        self.I = torch.prod(self.Is).item()

        P_mat = ten2mat(P, self.I)
        P_mat = normalize_rows(P_mat)
        self._mcm = MarkovChainMatrix(P_mat)

        self.P = mat2ten(self._mcm.P, self.Is)
        self.Q = mat2ten(self._mcm.Q, self.Is)
        self.R = self._mcm.R.reshape(tuple(self.Is))
        self.current_state = torch.tensor(np.unravel_index(self._mcm.current_state, tuple(self.Is)))

    def reset(self):
        self._mcm.reset()
        self.current_state = torch.tensor(np.unravel_index(self._mcm.current_state, tuple(self.Is)))

    def step(self):
        idx = self._mcm.step()
        self.current_state = torch.tensor(np.unravel_index(idx, tuple(self.Is)))
        return self.current_state

    def to_matrix(self) -> MarkovChainMatrix:
        return self._mcm
