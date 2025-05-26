import torch
import numpy as np
from tensorly.cp_tensor import cp_to_tensor

from src.chains.models import MarkovChainMatrix, MarkovChainTensor
from src.utils import mat2ten, ten2mat, lowtri2mat


class MatrixGenerator:
    def lowrank(self, I: int, K: int) -> MarkovChainMatrix:
        U = torch.randn(I, K)
        U = (U * U) / (torch.linalg.norm(U, dim=1, keepdim=True) ** 2)

        V = torch.randn(I, K)
        V = (V * V) / (torch.linalg.norm(V, dim=0, keepdim=True) ** 2)

        B = torch.diag(torch.FloatTensor(np.random.rand(K)))
        P = U @ B @ V.T
        P = P / P.sum(dim=1, keepdim=True)

        return MarkovChainMatrix(P)

    def erdosrenyi(self, I: int, edge_prob=0.3, eps=0.02, beta=1.0) -> MarkovChainMatrix:
        while True:
            A = lowtri2mat(np.random.binomial(1, edge_prob, int(I * (I - 1) / 2)))
            L = np.diag(A.sum(0)) - A
            if np.sum(np.abs(np.linalg.eigvalsh(L)) < 1e-9) <= 1:
                break

        At = A + beta * np.eye(I)
        P0 = At * (1 - eps) + eps
        P = torch.tensor(P0 / P0.sum(1, keepdims=True)).float()

        return MarkovChainMatrix(P)

    def block(self, I_per_block: int, K: int) -> MarkovChainMatrix:
        state_mat = torch.rand((K, K))
        Q = torch.kron(state_mat, torch.ones((I_per_block // K, I_per_block // K)))
        Q = Q / Q.sum(dim=1, keepdim=True)

        return MarkovChainMatrix(Q)


class TensorGenerator:
    def lowrank(self, Is: torch.Tensor, K: int) -> MarkovChainTensor:
        D = len(Is)
        I = torch.prod(Is).item()

        U = [torch.randn(Is[d % D], K) for d in range(2 * D)]
        U = [(u * u) / (torch.linalg.norm(u, dim=0, keepdim=True) ** 2) for u in U]
        w = torch.rand(K)
        w = w / w.sum()

        P = cp_to_tensor((w, U))
        P = ten2mat(P, I)
        P = P / P.sum(dim=1, keepdim=True)
        P = mat2ten(P, Is)

        return MarkovChainTensor(P)

    def block(self, I_per_block: int, D: int, K: int) -> MarkovChainTensor:
        state_core = torch.rand((K,) * (2 * D))
        Q = torch.kron(state_core, torch.ones((I_per_block // K,) * (2 * D)))
        Is = torch.tensor([I_per_block] * D)
        Q = Q / Q.sum(dim=tuple(range(D, 2 * D)), keepdim=True)

        return MarkovChainTensor(Q)

    def random_cp(self, Is: torch.Tensor, K: int):
        D = len(Is)
        weights = torch.rand(K)
        weights = weights / weights.sum()

        factors = []
        for d in range(D):
            factor = torch.rand(Is[d], K)
            factor = factor / factor.sum(dim=0, keepdim=True)
            factors.append(factor)

        return cp_to_tensor((weights, factors)), factors, weights
