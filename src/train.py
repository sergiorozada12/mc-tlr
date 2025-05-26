import os
import wandb
import numpy as np
import torch
from omegaconf import OmegaConf
from joblib import Parallel, delayed

from src.utils import kld_err, normfrob_err, norml1_err, mat2ten
from src.chains.models import MarkovChainMatrix
from src.estimation.baselines import IPDC, SGSADMM, SLRM
from src.estimation.scpd import SCPD


class Trainer:
    def __init__(self, method_name: str, cfg, I: int):
        self.cfg = cfg
        self.I = I
        self.method_name = method_name.lower()

        self.method_args = self._get_method_args()
        self.method_instance = self._instantiate_method()
        self.trial_results = []

        if self.cfg.general.use_wandb:
            wandb.init(
                project="markov-chain-estimation",
                config=OmegaConf.to_container(cfg, resolve=True),
                name=f"{method_name}_{self.method_args.get('K', 'NA')}",
            )

    def fit(self, trajectories, mc_true, mc_emp):
        num_cpus = os.cpu_count() // 2

        def run_trial(X, mc_emp_trial):
            if self.method_name in {"dc", "nn"}:
                P_emp = mc_emp_trial.P
                result = self.method_instance.fit(P_emp)
            elif self.method_name == "sm":
                Q_emp = mc_emp_trial.Q
                result = self.method_instance.fit(Q_emp)
            elif self.method_name in {"fib", "ent", "traj"}:
                Q_emp = mc_emp_trial.Q
                result = self.method_instance.fit(X, Q_emp)
            else:
                raise ValueError(f"Unknown method: {self.method_name}")

            mc_est = result["mc_est"]
            diffs = result.get("diffs", None)
            costs = result.get("costs", None)
            return mc_est, diffs, costs

        results = Parallel(n_jobs=num_cpus)(
            delayed(run_trial)(trajectories[t], mc_emp[t]) for t in range(self.cfg.general.trials)
        )
        self.trial_results = results
        self._log_all(mc_true)

    def _instantiate_method(self):
        method = self.method_name
        args = self.method_args

        if method == "dc":
            return IPDC(K=self.cfg.method.K, **args)
        elif method == "nn":
            return SGSADMM(K=self.cfg.method.K, **args)
        elif method == "sm":
            return SLRM(K=self.cfg.method.K, **args)
        elif method in {"fib", "ent", "traj"}:
            return SCPD(K=self.cfg.method.K, sampling_type=method, **args)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _get_method_args(self):
        method_cfg = getattr(self.cfg.method, self.method_name)
        method_args = {k: v for k, v in vars(method_cfg).items() if not k.startswith("_")}
        return method_args

    def _log_all(self, mc_true):
        trials = self.cfg.general.trials
        diffs, costs = zip(*[(r[1], r[2]) for r in self.trial_results])
        mc_ests = [r[0] for r in self.trial_results]

        klds = [kld_err(mc_true[t].P, mc_ests[t].P) for t in range(trials)]
        frobs = [normfrob_err(mc_true[t].P, mc_ests[t].P) for t in range(trials)]
        l1s = [norml1_err(mc_true[t].P, mc_ests[t].P) for t in range(trials)]

        wandb.log({
            "kld_mean": np.mean(klds),
            "frob_mean": np.mean(frobs),
            "l1_mean": np.mean(l1s),
        })

        for t in range(trials):
            if diffs[t] is not None:
                wandb.log({f"diffs/trial_{t}": wandb.plot.line_series(
                    xs=list(range(len(diffs[t]))),
                    ys=[diffs[t]],
                    keys=["diffs"],
                    title=f"Diffs Trial {t}",
                    xname="Iteration"
                )})
            if costs[t] is not None:
                wandb.log({f"costs/trial_{t}": wandb.plot.line_series(
                    xs=list(range(len(costs[t]))),
                    ys=[costs[t]],
                    keys=["costs"],
                    title=f"Costs Trial {t}",
                    xname="Iteration"
                )})

        avg_P = torch.stack([mc.P for mc in mc_ests]).mean(dim=0).numpy()
        wandb.log({"Estimated P (mean)": wandb.Image(avg_P)})

    def get_results(self):
        return self.trial_results
