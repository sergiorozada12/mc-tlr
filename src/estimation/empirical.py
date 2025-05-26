import torch
from typing import List, Tuple


class EmpiricalEstimator:
    def estimate_matrix(self, X: List[int], I: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert set(X) <= set(range(I)), "State indices out of bounds."

        N = len(X)
        pairwise_counts = torch.zeros((I, I))
        for t in range(N - 1):
            pairwise_counts[X[t], X[t + 1]] += 1

        marginal_counts = pairwise_counts.sum(dim=1, keepdim=True)
        mask = (marginal_counts == 0).expand_as(pairwise_counts)

        Q = pairwise_counts / N
        P = pairwise_counts / marginal_counts
        P[mask] = 1. / I

        return P, Q

    def estimate_tensor(self, X: List[List[int]], Is: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        D = len(Is)
        assert all(len(x) == D for x in X), "Each state must have D dimensions."

        I = torch.prod(Is).item()
        N = len(X)

        shape = tuple(Is.tolist()) * 2
        pairwise_counts = torch.zeros(shape)

        for t in range(N - 1):
            idx_from = tuple(X[t])
            idx_to = tuple(X[t + 1])
            pairwise_counts[idx_from + idx_to] += 1

        marginal_counts = pairwise_counts.sum(dim=tuple(range(D, 2 * D)), keepdim=True)
        mask = (marginal_counts == 0).expand_as(pairwise_counts)

        Q = pairwise_counts / N
        P = pairwise_counts / marginal_counts
        P[mask] = 1. / I

        return P, Q

    def estimate_matrix_batch(self, trajectories: List[List[int]], I: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [self.estimate_matrix(x, I) for x in trajectories]

    def estimate_tensor_batch(self, trajectories: List[List[List[int]]], Is: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [self.estimate_tensor(x, Is) for x in trajectories]
