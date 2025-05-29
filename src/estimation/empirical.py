import torch
import numpy as np
from typing import List, Tuple


class EmpiricalEstimator:
    def estimate_matrix(
        self, X: List[int], I: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert set(X) <= set(range(I)), "State indices out of bounds."
        N = len(X)

        pairwise_counts = torch.zeros((I, I))
        X_all_steps = np.array([np.array(X)[:-1], np.array(X)[1:]])
        X_steps, counts = np.unique(X_all_steps, axis=1, return_counts=True)
        pairwise_counts[X_steps[0], X_steps[1]] = torch.tensor(
            counts, dtype=torch.float
        )

        marginal_counts = pairwise_counts.sum(dim=1, keepdim=True)
        mask = (marginal_counts == 0).expand_as(pairwise_counts)

        Q = pairwise_counts / N
        P = pairwise_counts / marginal_counts
        P[mask] = 1.0 / I

        return P, Q

    def estimate_tensor(
        self, X: List[List[int]], Is: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        D = len(Is)
        assert all(len(x) == D for x in X), "Each state must have D dimensions."
        I = torch.prod(Is).item()
        N = len(X)

        shape = tuple(Is.tolist()) * 2
        pairwise_counts = torch.zeros(shape)

        X_all_steps = np.concatenate([np.array(X)[:-1], np.array(X)[1:]], axis=1).T
        X_steps, counts = np.unique(X_all_steps, axis=1, return_counts=True)

        for idx, count in zip(X_steps.T, counts):
            idx_from = tuple(idx[:D])
            idx_to = tuple(idx[D:])
            pairwise_counts[idx_from + idx_to] = float(count)

        marginal_counts = pairwise_counts.sum(dim=tuple(range(D, 2 * D)), keepdim=True)
        mask = (marginal_counts == 0).expand_as(pairwise_counts)

        Q = pairwise_counts / N
        P = pairwise_counts / marginal_counts
        P[mask] = 1.0 / I

        return P, Q

    def estimate_matrix_batch(
        self, trajectories: List[List[int]], I: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [self.estimate_matrix(x, I) for x in trajectories]

    def estimate_tensor_batch(
        self, trajectories: List[List[List[int]]], Is: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [self.estimate_tensor(x, Is) for x in trajectories]
