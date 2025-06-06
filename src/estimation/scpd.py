import torch
import numpy as np
from time import perf_counter
from tensorly.cp_tensor import cp_to_tensor
from tensorly.tenalg import khatri_rao
from tensorly import unfold

from src.chains.models import MarkovChainTensor
from src.utils import (
    proj_bounded_simplex,
    norml1_err,
    kld_err,
    qten_to_pten,
    valid_joint_ten,
    valid_joint_mat,
    mat_to_ten,
    ten_to_mat,
)


class SCPD:
    def __init__(
        self,
        K: int,
        qmin: float = 0.0,
        qmax: float = 1.0,
        sampling_type: str = "trajectory",
        alpha_type: str = "adam",
        alpha_factor: float = 1.0,
        alpha_weight: float = 1.0,
        gamma_factor: float = 0.0,
        gamma_weight: float = 0.0,
        beta: float = 1e-1,
        eps: float = 1e-9,
        B: int = 10,
        B_max: int = None,
        increase_B: bool = False,
        acceleration: bool = False,
        tol: float = 1e-8,
        num_itrs: int = 1000,
        slide_window: int = 50,
        verbose: bool = False,
    ):
        self.K = K
        self.qmin = qmin
        self.qmax = qmax
        self.sampling_type = sampling_type or "trajectory"
        self.alpha_type = alpha_type or "constant"
        self.alpha_factor_init = alpha_factor
        self.alpha_weight_init = alpha_weight
        self.gamma_factor = gamma_factor
        self.gamma_weight = gamma_weight
        self.beta = beta
        self.eps = eps
        self.B = B
        self.B_max = B_max or np.inf
        self.increase_B = increase_B
        self.acceleration = acceleration
        self.tol = tol
        self.num_itrs = num_itrs
        self.slide_window = slide_window
        self.verbose = verbose

    def fit(self, chain, Q_emp, Is=None):
        assert valid_joint_ten(Q_emp) or valid_joint_mat(Q_emp), "Invalid joint probabilities. Expecting valid probability tensor or matrix."
        if Is is None:
            assert valid_joint_ten(Q_emp), "Invalid joint probability tensor. Expecting tensor with an even number of dimensions with entries in [0,1] that sum to 1."
            DD = Q_emp.ndim
            D = DD // 2
            IIs = torch.tensor(Q_emp.shape)
            Is = IIs[:D]
            II = IIs.prod().item()
        else:
            assert torch.tensor(Q_emp.shape).prod()==Is.prod()**2, "Inconsistent number of states between empirical tensor and given dimensions."
            if valid_joint_mat(Q_emp) and Is.ndim>1:
                Q_emp = mat_to_ten(Q_emp,Is)
            elif valid_joint_ten(Q_emp):
                I = int(torch.tensor(Q_emp.shape).prod().sqrt())
                Q_emp = mat_to_ten(ten_to_mat(Q_emp,I),Is)
            D = len(Is)
            DD = 2*D
            IIs = Is.repeat(2)
            II = IIs.prod().item()

        Q_est, Qds, l = self._init_tensor(IIs)
        Qds_upd = [Q.clone() for Q in Qds]
        l_upd = l.clone()

        obs_pairs = (
            torch.where(Q_emp.flatten() != 0)[0]
            if self.sampling_type == "entry"
            else None
        )
        R_emp = (
            0.5 * (Q_emp.sum(tuple(range(D, 2 * D))) + Q_emp.sum(tuple(range(D))))
            if self.sampling_type == "trajectory"
            else None
        )
        chain_transitions = (
            torch.hstack([torch.stack(chain[1:]), torch.stack(chain[:-1])])
            if self.sampling_type == "trajectory"
            else None
        )

        diffs, costs, variances = [], [], []
        alpha_factor, alpha_weight = self.alpha_factor_init, self.alpha_weight_init
        factor_grad_mags = torch.zeros(DD)
        weight_grad_mag = 0.0
        mttkrp = int(np.ceil(II / self.B / IIs.min().item()))
        num_mttkrps = int(self.num_itrs / mttkrp + 0.5)
        B_init = self.B

        t, tot_itr = 0.0, 0
        TERMINATE = False
        tic = perf_counter()

        for out_itr in range(num_mttkrps):
            for inn_itr in range(mttkrp):
                Q_last = Q_est.clone()
                t_last = t
                t = 0.5 * (1 + np.sqrt(1 + 4 * t_last**2))

                d = np.random.choice(DD + 1)

                if self.increase_B:
                    B_new = int(np.ceil((out_itr + 1) ** 0.5))
                    self.B = min(max(B_new, B_init), self.B_max)

                if d < DD:
                    Gd = self._factor_compute_grad(
                        d,
                        chain,
                        Q_emp,
                        Qds_upd,
                        l,
                        Q_est,
                        obs_pairs,
                        R_emp,
                        chain_transitions,
                    )
                    factor_grad_mags[d] += torch.linalg.norm(Gd.reshape(-1), 1)
                    Qd_last = Qds[d].clone()
                    Q_dot, alpha_factor = self._grad_step(
                        Qds[d], Gd, alpha_factor, factor_grad_mags[d]
                    )
                    Qds[d] = self._proj_factor(Q_dot)
                    Qds_upd[d] = (
                        Qds[d] + (1 - t_last) / t * (Qds[d] - Qd_last)
                        if self.acceleration
                        else Qds[d].clone()
                    )
                else:
                    gl = self._weight_compute_grad(
                        chain, Q_emp, Qds, l_upd, obs_pairs, R_emp, chain_transitions
                    )
                    weight_grad_mag += torch.linalg.norm(gl.reshape(-1), 1)
                    l_last = l.clone()
                    l_dot, alpha_weight = self._grad_step(
                        l, gl, alpha_weight, weight_grad_mag
                    )
                    l = self._proj_weight(l_dot)
                    l_upd = (
                        l + (1 - t_last) / t * (l - l_last)
                        if self.acceleration
                        else l.clone()
                    )

                Q_est = cp_to_tensor((l, Qds))
                diff = norml1_err(Q_est, Q_last)
                cost = kld_err(Q_emp, Q_est)
                var = (
                    np.var(costs[tot_itr - self.slide_window : tot_itr])
                    if tot_itr >= self.slide_window
                    else 0.0
                )

                diffs.append(diff)
                costs.append(cost)
                variances.append(var)

                toc = perf_counter()
                if self.verbose and (out_itr == 0 and inn_itr == 0) or (toc - tic > 2):
                    print(
                        f"MTTKRP: {out_itr+1}/{num_mttkrps} | Cost: {cost:.3e} | Diff: {diff:.1e} | Var: {var:.1e}"
                    )
                    tic = perf_counter()

                if np.isnan(cost) or torch.allclose(Q_est, torch.zeros_like(Q_est)):
                    if self.verbose:
                        print("Invalid solution. Returning null.")
                    return None

                if tot_itr >= self.slide_window and var < self.tol:
                    if self.verbose:
                        print(f"Terminating early @ {out_itr+1}. Variance = {var:.1e}")
                    TERMINATE = True
                    break

                tot_itr += 1

            if TERMINATE:
                break

        Q_est = cp_to_tensor((l, Qds))
        P_est = qten_to_pten(Q_est)
        return dict(
            mc_est=MarkovChainTensor(P_est), diffs=diffs, costs=costs, Qds=Qds, l=l
        )

    def _init_tensor(self, shape):
        D = len(shape)
        factors = []
        for d in range(D):
            if shape[d] == 0:
                raise ValueError(f"Invalid dimension: shape[{d}] = 0")
            f = torch.rand(shape[d], self.K)
            f = f / f.sum(dim=0, keepdim=True)
            factors.append(f)
        weights = torch.rand(self.K)
        weights = weights / weights.sum()
        Q_est = cp_to_tensor((weights, factors))
        return Q_est, factors, weights

    def _grad_step(self, X, G, alpha, grad_mag):
        step = self._update_alpha(alpha, grad_mag)
        X_dot = (X - step * G).numpy().astype(float)
        while all(X_dot.flatten() <= 0.0):
            alpha *= 0.9
            step = self._update_alpha(alpha, grad_mag)
            X_dot = (X - step * G).numpy().astype(float)
        return X_dot, alpha

    def _update_alpha(self, alpha, grad_mag):
        if self.alpha_type == "decay":
            return alpha / (1 + grad_mag)
        elif self.alpha_type == "adam":
            return alpha / (grad_mag**self.beta + self.eps)
        else:
            return alpha

    def _proj_factor(self, Qd):
        s = float((1 - Qd.shape[0] * self.qmin) / (self.qmax - self.qmin))
        proj = [
            proj_bounded_simplex((Qd[:, k] - self.qmin) / (self.qmax - self.qmin), s)
            for k in range(Qd.shape[1])
        ]
        return self.qmin + (self.qmax - self.qmin) * torch.FloatTensor(np.array(proj).T)

    def _proj_weight(self, l):
        return self.qmin + (self.qmax - self.qmin) * torch.FloatTensor(
            proj_bounded_simplex(
                (l - self.qmin) / (self.qmax - self.qmin),
                (1 - len(l) * self.qmin) / (self.qmax - self.qmin),
            )
        )

    def _weight_compute_grad(
        self, chain, Q_emp, Qds, l, obs_pairs, R_emp, chain_transitions
    ):
        DD = Q_emp.ndim
        D = DD // 2
        IIs = torch.tensor(Q_emp.shape)
        Is = IIs[:D]
        II = IIs.prod().item()
        I = Is.prod().item()

        if self.sampling_type == "fib":
            B = np.minimum(self.B, int(II))
            fibs = np.random.choice(int(II), B, replace=False)

            ml = Q_emp.reshape(-1)[fibs]
            Hl = khatri_rao(Qds)[fibs]
            gl = -(ml / (Hl @ l + self.eps)) @ Hl / B

            if self.gamma_weight > 0:
                B_erg = int(np.minimum(B, I))
                fibs_erg = np.random.choice(I, B_erg, replace=False)
                H0_erg = khatri_rao(Qds[:D])[fibs_erg]
                H1_erg = khatri_rao(Qds[D:])[fibs_erg]
                gl_erg = (
                    self.gamma_weight
                    * ((H0_erg - H1_erg).T @ (H0_erg - H1_erg))
                    @ l
                    / B_erg
                )
                gl += gl_erg

        elif self.sampling_type == "ent":
            if obs_pairs is None:
                obs_pairs = torch.where(Q_emp.flatten() != 0)[0]
            B = np.minimum(self.B, len(obs_pairs))
            inds = np.random.choice(obs_pairs, B, replace=False)
            steps = [torch.tensor(np.unravel_index(step, tuple(IIs))) for step in inds]

            hls = [
                torch.stack([Qds[d][step[d]] for d in range(DD)]).prod(0)
                for step in steps
            ]
            gls = [
                -(Q_emp[*step] * II) * hls[i] / (hls[i] @ l + self.eps)
                for i, step in enumerate(steps)
            ]
            gl = torch.stack(gls).sum(0) / B

            if self.gamma_weight > 0:
                B_erg = np.minimum(B, len(chain))
                steps_erg = torch.stack(chain)[
                    np.random.choice(len(chain), B_erg, replace=False)
                ]

                # I think next line is wrong, chain is a collection of states, not transitions, so it is not needed
                # steps_erg = [step[:D] if np.random.rand() < .5 else step[D:] for step in steps_erg]

                hs0 = [
                    torch.stack([Qds[d][step[d % D]] for d in range(D)]).prod(0)
                    for step in steps_erg
                ]
                hs1 = [
                    torch.stack([Qds[d][step[d % D]] for d in range(D, DD)]).prod(0)
                    for step in steps_erg
                ]
                gls_erg = [
                    self.gamma_weight
                    * (II * I)
                    * torch.outer(hs0[i] - hs1[i], hs0[i] - hs1[i])
                    @ l
                    for i in range(B_erg)
                ]
                gl_erg = torch.stack(gls_erg).sum(0) / B_erg
                gl += gl_erg

        elif self.sampling_type == "traj":
            if R_emp is None:
                R_emp = 0.5 * (
                    Q_emp.sum(tuple(range(D, 2 * D))) + Q_emp.sum(tuple(range(D)))
                )
            if chain_transitions is None:
                chain_transitions = torch.hstack(
                    [torch.stack(chain[1:]), torch.stack(chain[:-1])]
                )

            B = np.minimum(self.B, len(chain) - 1)
            inds = np.random.choice(len(chain) - 1, B, replace=False)
            steps = chain_transitions[inds]

            hls = [
                torch.stack([Qds[d][step[d]] for d in range(DD)]).prod(0)
                for step in steps
            ]
            gls = [-hls[i] / (hls[i] @ l + self.eps) for i, step in enumerate(steps)]
            gl = torch.stack(gls).sum(0) / B

            if self.gamma_weight > 0:
                B_erg = np.minimum(B, len(chain))
                steps_erg = torch.stack(chain)[
                    np.random.choice(len(chain), B_erg, replace=False)
                ]

                # I think next line is wrong, chain is a collection of states, not transitions, so it is not needed
                # steps_erg = [step[:D] if np.random.rand() < .5 else step[D:] for step in steps_erg]

                hs0 = [
                    torch.stack([Qds[d][step[d % D]] for d in range(D)]).prod(0)
                    for step in steps_erg
                ]
                hs1 = [
                    torch.stack([Qds[d][step[d % D]] for d in range(D, DD)]).prod(0)
                    for step in steps_erg
                ]
                gls_erg = [
                    self.gamma_weight
                    / R_emp[*step]
                    * torch.outer(hs0[i] - hs1[i], hs0[i] - hs1[i])
                    @ l
                    for i, step in enumerate(steps_erg)
                ]
                gl_erg = torch.stack(gls_erg).sum(0) / B_erg
                gl += gl_erg
            else:
                raise f"Sampling method {self.sampling_type} not defined"

        return gl

    def _factor_compute_grad(
        self, d, chain, Q_emp, Qds, l, Q_est, obs_pairs, R_emp, chain_transitions
    ):
        DD = Q_emp.ndim
        D = DD // 2
        IIs = torch.tensor(Q_emp.shape)
        Is = IIs[:D]
        II = IIs.prod().item()
        I = Is.prod().item()
        assert d in np.arange(DD), "Invalid dimension d."

        if self.sampling_type == "fib":
            B = np.minimum(self.B, int(II / IIs[d]))
            fibs = np.random.choice(int(II / IIs[d]), B, replace=False)

            Md = unfold(Q_emp, mode=d).T[fibs]
            Hd = khatri_rao([Qds[i] for i in range(DD) if i != d])[fibs]
            L = (Md / (Hd @ torch.diag(l) @ Qds[d].T + self.eps)) * (Md > 0)
            Gd = -L.T @ Hd @ torch.diag(l) / B

            if self.gamma_factor > 0:
                B_erg = int(np.minimum(B, int(I / IIs[d])))
                fibs_erg = np.random.choice(int(I / IIs[d]), B_erg, replace=False)

                Q0_erg = unfold(cp_to_tensor((l, Qds[:D])), mode=d % D).T[fibs_erg] * (
                    2 * (d < D) - 1
                )
                Q1_erg = unfold(cp_to_tensor((l, Qds[D:])), mode=d % D).T[fibs_erg] * -(
                    2 * (d < D) - 1
                )
                H_erg = khatri_rao(
                    [Qds[i] for i in np.arange(D) + D * (d >= D) if i != d]
                )[fibs_erg]
                Gd_erg = (
                    self.gamma_factor
                    * (Q0_erg + Q1_erg).T
                    @ H_erg
                    @ torch.diag(l)
                    / B_erg
                )
                Gd += Gd_erg

        elif self.sampling_type == "ent":
            if Q_est is None:
                Q_est = cp_to_tensor((l, Qds))
            if obs_pairs is None:
                obs_pairs = torch.where(Q_emp.flatten() != 0)[0]

            B = np.minimum(self.B, len(obs_pairs))
            inds = np.random.choice(obs_pairs, B, replace=False)
            steps = [torch.tensor(np.unravel_index(step, tuple(IIs))) for step in inds]

            hds = [
                torch.stack([Qds[j][step[j]] for j in range(DD) if j != d]).prod(0)
                for step in steps
            ]
            gds = [
                -(Q_emp[*step] * II) * hds[i] * l / (Q_est[*step] + self.eps)
                for i, step in enumerate(steps)
            ]
            Gds = [
                torch.outer(torch.eye(IIs[d])[step[d]], gds[i])
                for i, step in enumerate(steps)
            ]
            Gd = torch.stack(Gds).sum(0) / B

            if self.gamma_factor > 0:
                steps_erg = [step[:D] if d < D else step[D:] for step in steps]

                Q0 = cp_to_tensor((l, Qds[:D])) * (2 * (d < D) - 1)
                Q1 = cp_to_tensor((l, Qds[D:])) * -(2 * (d < D) - 1)
                hds_erg = [
                    torch.stack(
                        [
                            Qds[j][step[j % D]]
                            for j in np.arange(D) + D * (d >= D)
                            if j != d
                        ]
                    ).prod(0)
                    for step in steps_erg
                ]
                gds_erg = [
                    self.gamma_factor
                    * (II * I)
                    * (Q0[*step] + Q1[*step])
                    * hds_erg[i]
                    * l
                    for i, step in enumerate(steps_erg)
                ]
                Gds_erg = [
                    torch.outer(torch.eye(IIs[d])[step[d % D]], gds_erg[i])
                    for i, step in enumerate(steps_erg)
                ]
                Gd_erg = torch.stack(Gds_erg).sum(0) / B
                Gd += Gd_erg

        elif self.sampling_type == "traj":
            if Q_est is None:
                Q_est = cp_to_tensor((l, Qds))
            if R_emp is None:
                R_emp = 0.5 * (
                    Q_emp.sum(tuple(range(D, 2 * D))) + Q_emp.sum(tuple(range(D)))
                )
            if chain_transitions is None:
                chain_transitions = torch.hstack(
                    [torch.stack(chain[1:]), torch.stack(chain[:-1])]
                )

            B = np.minimum(self.B, len(chain) - 1)
            inds = np.random.choice(len(chain) - 1, B, replace=False)
            steps = chain_transitions[inds]

            hds = [
                torch.stack([Qds[j][step[j]] for j in range(DD) if j != d]).prod(0)
                for step in steps
            ]
            gds = [
                -hds[i] * l / (Q_est[*step] + self.eps) for i, step in enumerate(steps)
            ]
            Gds = [
                torch.outer(torch.eye(IIs[d])[step[d]], gds[i])
                for i, step in enumerate(steps)
            ]
            Gd = torch.stack(Gds).sum(0) / B

            if self.gamma_factor > 0:
                steps_erg = [step[:D] if d < D else step[D:] for step in steps]

                Q0 = cp_to_tensor((l, Qds[:D])) * (2 * (d < D) - 1)
                Q1 = cp_to_tensor((l, Qds[D:])) * -(2 * (d < D) - 1)
                hds_erg = [
                    torch.stack(
                        [
                            Qds[j][step[j % D]]
                            for j in np.arange(D) + D * (d >= D)
                            if j != d
                        ]
                    ).prod(0)
                    for step in steps_erg
                ]
                gds_erg = [
                    self.gamma_factor
                    / R_emp[*step]
                    * (Q0[*step] + Q1[*step])
                    * hds_erg[i]
                    * l
                    for i, step in enumerate(steps_erg)
                ]
                Gds_erg = [
                    torch.outer(torch.eye(IIs[d])[step[d % D]], gds_erg[i])
                    for i, step in enumerate(steps_erg)
                ]
                Gd_erg = torch.stack(Gds_erg).sum(0) / B
                Gd += Gd_erg
        else:
            raise f"Sampling method {self.sampling_type} not defined"

        return Gd
