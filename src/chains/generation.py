import torch
import numpy as np
from tensorly.cp_tensor import cp_to_tensor

from src.chains.models import MarkovChainMatrix, MarkovChainTensor
from src.utils import mat_to_ten, ten_to_mat, lowtri_to_mat


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

    def erdosrenyi(
        self, I: int, edge_prob=0.3, eps=0.02, beta=1.0
    ) -> MarkovChainMatrix:
        while True:
            A = lowtri_to_mat(np.random.binomial(1, edge_prob, int(I * (I - 1) / 2)))
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
        P = ten_to_mat(P, I)
        P = P / P.sum(dim=1, keepdim=True)
        P = mat_to_ten(P, Is)

        return MarkovChainTensor(P), U, w

    # def block(self, num_blocks: int, I_per_block: int, K_per_block: int, D: int) -> MarkovChainTensor:
    def block(self, Is: torch.tensor, K: int) -> MarkovChainTensor:
        assert K>Is[0], "Invalid dimensions. Tensor rank must be larger than number of blocks."
        D = len(Is)
        I = Is.prod().item()

        num_blocks = Is[0].item()
        block_labs = np.random.choice(num_blocks,K)
        while set(np.unique(block_labs)) != set(np.arange(num_blocks)):
            block_labs = np.random.choice(num_blocks,K)
        block_labs = np.sort(block_labs)
        block_inds = [np.where(block_labs!=k)[0] for k in range(num_blocks)]
        # K_per_block = int(K/num_blocks)
        # block_inds = np.concatenate(([np.arange(num_blocks)*K_per_block,[K]]))

        U = [(torch.randn(Is[d%D],K))**2 for d in range(2*D)]
        for k in range(num_blocks):
            U[0][k,block_inds[k]] *= 1e-4
            U[D][k,block_inds[k]] *= 1e-4
        U = [u/torch.linalg.norm(u,dim=0,keepdim=True) for u in U]
        w = torch.rand(K); w = w/w.sum()

        Pten = cp_to_tensor((w,U))
        P = ten_to_mat(Pten,I)
        marg = P.sum(dim=1)
        P[marg!=0] = P[marg!=0]/marg[marg!=0][:,None]
        return MarkovChainTensor(mat_to_ten(P,Is)), U, w

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
