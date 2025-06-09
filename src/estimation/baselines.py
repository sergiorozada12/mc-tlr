import torch
import numpy as np
from time import perf_counter
from scipy.sparse.linalg import svds
from src.chains.models import MarkovChainMatrix
from src.utils import (
    kld_err,
    qmat_to_pmat,
    valid_transition_mat,
    valid_joint_mat,
)


class SGSADMM:
    def __init__(
        self,
        beta=1.0,
        gamma=1.0,
        pmin=0.0,
        num_itrs=5000,
        tol=1e-6,
        K=None,
        verbose=False,
    ):
        self.beta = beta
        self.gamma = gamma
        self.pmin = pmin
        self.num_itrs = num_itrs
        self.tol = tol
        self.K = K
        self.verbose = verbose

    def fit(self, P_emp):
        assert valid_transition_mat(P_emp), "Invalid transition probability matrix. Expecting a right-stochastic square matrix with entries in [0,1]."
        I = P_emp.shape[0]
        pmin_tensor = torch.tensor(self.pmin, device=P_emp.device, dtype=P_emp.dtype)

        def update_y(P, W, S):
            return (torch.eye(I, device=P.device) - P - self.beta * (W + S)).sum(1) / (
                self.beta * I
            )

        def update_W(P, S, y):
            R = torch.outer(y, torch.ones(I, device=P.device)) + S + P / self.beta
            Z = 0.5 * (R + torch.sqrt(R**2 + 4 * P_emp / self.beta)) * (
                P_emp != 0
            ) + torch.maximum(R, torch.zeros_like(R)) * (P_emp == 0)
            return Z - R

        def update_S(P, W, y):
            R = -(W + torch.outer(y, torch.ones(I, device=P.device)) + P / self.beta)
            try:
                if self.K is None:
                    U, sig, V = torch.svd(R)
                else:
                    U, sig, V = torch.svd_lowrank(R, self.K)
            except:
                U, sig, V = np.linalg.svd(R.cpu().numpy())
                U = torch.FloatTensor(U).to(P.device)
                sig = torch.FloatTensor(sig).to(P.device)
                V = torch.FloatTensor(V).to(P.device)
            sig_upd = torch.minimum(
                sig, torch.tensor(self.gamma, device=sig.device, dtype=sig.dtype)
            )
            return U @ torch.diag(sig_upd) @ V.T

        def update_P(P, W, S, y):
            return torch.maximum(
                P
                + self.gamma
                * self.beta
                * (W + torch.outer(y, torch.ones(I, device=P.device)) + S),
                pmin_tensor,
            )

        P = torch.rand(I, I, device=P_emp.device)
        P = P / P.sum(1, keepdim=True)
        W = torch.zeros((I, I), device=P.device)
        S = torch.zeros((I, I), device=P.device)
        y = -torch.ones(I, device=P.device)

        diffs = []
        costs = []

        tic = perf_counter()
        for itr in range(self.num_itrs):
            y_last, W_last, S_last, P_last = y.clone(), W.clone(), S.clone(), P.clone()

            y = update_y(P, W, S)
            W = update_W(P, S, y)
            y = update_y(P, W, S)
            S = update_S(P, W, y)
            P = update_P(P, W, S, y)

            diff = (
                torch.tensor(
                    [
                        torch.norm(y - y_last),
                        torch.norm(W - W_last),
                        torch.norm(S - S_last),
                        torch.norm(P - P_last),
                    ]
                )
                .max()
                .item()
            )
            cost = kld_err(P_emp, P)

            diffs.append(diff)
            costs.append(cost)

            toc = perf_counter()
            if self.verbose and toc - tic > 5:
                print(
                    f"Iter. {itr+1}/{self.num_itrs} | Cost: {cost:.1e} | Diff: {diff:.1e}"
                )
                tic = perf_counter()

            if self.verbose and diff < self.tol:
                print(f"Terminating early @ {itr+1} iters. Diff = {diff:.2e}")
                break

        P = P / P.sum(1, keepdim=True)
        return dict(mc_est=MarkovChainMatrix(P), diffs=diffs, costs=costs)


class IPDC:
    def __init__(
        self,
        K,
        beta=1.0,
        gamma=1.0,
        alpha=1e-1,
        pmin=0.0,
        num_itrs=5000,
        tol=1e-6,
        num_inn_itrs=1,
        inn_tol=1e-6,
        verbose=False,
    ):
        self.K = K
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.pmin = pmin
        self.num_itrs = num_itrs
        self.tol = tol
        self.num_inn_itrs = num_inn_itrs
        self.inn_tol = inn_tol
        self.verbose = verbose

    def fit(self, P_emp):
        assert valid_transition_mat(P_emp), "Invalid transition probability matrix. Expecting a right-stochastic square matrix with entries in [0,1]."
        I = P_emp.shape[0]
        pmin_tensor = torch.tensor(self.pmin, device=P_emp.device, dtype=P_emp.dtype)

        def update_y(P, W, S):
            return (torch.eye(I, device=P.device) - P - self.beta * (W + S)).sum(1) / (
                self.beta * I
            )

        def update_W(P, S, T, y):
            R = torch.outer(y, torch.ones(I, device=P.device)) + S + P / self.beta
            sqrt_term = torch.sqrt(
                (R - T / self.beta) ** 2 + 4 * (self.alpha + 1) * P_emp / self.beta
            )
            Z = (0.5 / (self.alpha + 1)) * ((R - T / self.beta) + sqrt_term) * (
                P_emp != 0
            ) + torch.maximum(R - T / self.beta, torch.zeros_like(R)) * (P_emp == 0)
            return Z - R

        def update_S(P, W, y):
            R = -(W + torch.outer(y, torch.ones(I, device=P.device)) + P / self.beta)
            try:
                U, sig, V = torch.svd_lowrank(R, self.K)
            except:
                U, sig, V = np.linalg.svd(R.cpu().numpy())
                U = torch.FloatTensor(U).to(P.device)
                sig = torch.FloatTensor(sig).to(P.device)
                V = torch.FloatTensor(V).to(P.device)
            sig_upd = torch.minimum(
                sig, torch.tensor(self.gamma, device=sig.device, dtype=sig.dtype)
            )
            return U @ torch.diag(sig_upd) @ V.T

        def update_T(P):
            try:
                U, sig, V = svds(P.cpu().numpy(), k=self.K)
                return torch.FloatTensor(U @ np.diag(sig) @ V).to(P.device)
            except:
                return P.clone()

        def update_P(P, W, S, y):
            return torch.maximum(
                P
                + self.gamma
                * self.beta
                * (W + torch.outer(y, torch.ones(I, device=P.device)) + S),
                pmin_tensor,
            )

        P = torch.ones((I, I), device=P_emp.device) / I
        W = torch.zeros((I, I), device=P.device)
        S = torch.zeros((I, I), device=P.device)
        y = -torch.ones(I, device=P.device)
        T = update_T(P)

        diffs = []
        costs = []

        tic = perf_counter()
        for itr in range(self.num_itrs):
            y_last, W_last, S_last, P_last, T_last = (
                y.clone(),
                W.clone(),
                S.clone(),
                P.clone(),
                T.clone(),
            )
            T = update_T(P)

            for _ in range(self.num_inn_itrs):
                y_inn = update_y(P, W, S)
                W = update_W(P, S, T, y_inn)
                y = update_y(P, W, S)
                S = update_S(P, W, y)
                P = update_P(P, W, S, y)

                if (
                    torch.max(
                        torch.tensor(
                            [
                                torch.norm(y - y_inn),
                                torch.norm(W - W_last),
                                torch.norm(S - S_last),
                                torch.norm(P - P_last),
                            ]
                        )
                    ).item()
                    < self.inn_tol
                ):
                    break

            diff = (
                torch.tensor(
                    [
                        torch.norm(y - y_last),
                        torch.norm(W - W_last),
                        torch.norm(S - S_last),
                        torch.norm(P - P_last),
                        torch.norm(T - T_last),
                    ]
                )
                .max()
                .item()
            )
            cost = kld_err(P_emp, P)

            diffs.append(diff)
            costs.append(cost)

            toc = perf_counter()
            if self.verbose and toc - tic > 5:
                print(
                    f"Iter. {itr+1}/{self.num_itrs} | Cost: {cost:.2e} | Diff: {diff:.2e}"
                )
                tic = perf_counter()

            if self.verbose and diff < self.tol:
                print(f"Terminating early @ {itr+1}. Diff = {diff:.2e}")
                break

        P = P / P.sum(1, keepdim=True)
        return dict(mc_est=MarkovChainMatrix(P), diffs=diffs, costs=costs)


class SLRM:
    def __init__(self, K, qmin=0.0):
        self.K = K
        self.qmin = qmin

    def fit(self, Q_emp):
        assert valid_joint_mat(Q_emp), "Invalid joint probability matrix. Expecting a square matrix with entries in [0,1] that sum to 1."
        I = Q_emp.shape[0]
        qmin_tensor = torch.tensor(self.qmin, device=Q_emp.device, dtype=Q_emp.dtype)

        try:
            U, sig, V = svds(Q_emp.cpu().numpy(), k=self.K)
            Q_est = torch.FloatTensor(U @ np.diag(sig) @ V).to(Q_emp.device)
        except:
            U, sig, V = np.linalg.svd(Q_emp.cpu().numpy())
            Q_est = torch.FloatTensor(
                U[:, : self.K] @ np.diag(sig[: self.K]) @ V[: self.K, :]
            ).to(Q_emp.device)

        Q_est = torch.maximum(Q_est, qmin_tensor)
        Q_est = Q_est / torch.linalg.norm(Q_est, 1)

        R = 0.5 * (Q_est.sum(0) + Q_est.sum(1))
        P = torch.diag(1 / (R + (R == 0).float())) @ Q_est
        Marginal = P.sum(1, keepdim=True)
        Mask = (Marginal == 0).expand_as(P)
        P = P / Marginal
        P[Mask] = torch.tensor(1.0 / I, device=P.device, dtype=P.dtype)

        P = P / P.sum(1, keepdim=True)
        return dict(mc_est=MarkovChainMatrix(P), cost=kld_err(qmat_to_pmat(Q_emp), P))
