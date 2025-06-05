import os
import wandb
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from joblib import Parallel, delayed

from src.utils import kld_err, normfrob_err, norml1_err, ten_to_mat, mat_to_ten
from src.estimation.baselines import IPDC, SGSADMM, SLRM
from src.estimation.scpd import SCPD

# TODO: Make saving results optional
# TODO: How to save factor matrices and normalization weights?

class Trainer:
    def __init__(self, project: str, experiment_name: str, cfg ):
        self.cfg = cfg
        self.project = project
        self.experiment_name = experiment_name
        self.method_name = cfg.method.method_name
        self.trial_results = []

        if self.cfg.general.use_wandb:
            wandb.init(
                project=self.project,
                config=OmegaConf.to_container(cfg, resolve=True),
                name=self.experiment_name,
            )
            self.method_name = wandb.config['method']['method_name']

        # Initialize method arguments after initializing wandb for values that change
        self.method_args = self._get_method_args()
        self.method_instance = self._instantiate_method()
    
    def fit(self, trajectories, mc_true, mc_emp, Is=None):
        num_cpus = os.cpu_count() // 2
        trials = len(mc_true)

        def run_trial(X, mc_emp_trial):
            if self.method_name in {"dc", "nn"}:
                P_emp = ten_to_mat(mc_emp_trial.P,int(torch.tensor(P_emp.shape).prod().sqrt())) if mc_emp_trial.P.ndim!=2 else mc_emp_trial.P
                result = self.method_instance.fit(P_emp)
            elif self.method_name == "sm":
                Q_emp = ten_to_mat(mc_emp_trial.Q,int(torch.tensor(Q_emp.shape).prod().sqrt())) if mc_emp_trial.Q.ndim!=2 else mc_emp_trial.Q
                result = self.method_instance.fit(Q_emp)
            elif self.method_name in {"fib", "ent", "traj"}:
                Q_emp = mat_to_ten(mc_emp_trial.Q, Is) if not all(torch.tensor(mc_emp_trial.Q.shape)==Is.repeat(2)) else mc_emp_trial.Q
                result = self.method_instance.fit(X, Q_emp, Is)
            else:
                raise ValueError(f"Unknown method: {self.method_name}")

            mc_est = result["mc_est"]
            diffs = result.get("diffs", None)
            costs = result.get("costs", None)
            return mc_est, diffs, costs

        results = Parallel(n_jobs=num_cpus)(
            delayed(run_trial)(trajectories[t], mc_emp[t]) for t in range(trials)
        )
        self._prepare_all_metrics(results, mc_true)
        self._log_all()
        self._save_results_as_json()

        # Values tracked for sweeps
        wandb.log({"qloss":float(np.mean(self.errors['kld']['Q']))})
        wandb.log({"ploss":float(np.mean(self.errors['kld']['P']))})

    def _instantiate_method(self):
        if self.method_name == "dc":
            return IPDC(**self.method_args)
        elif self.method_name == "nn":
            return SGSADMM(**self.method_args)
        elif self.method_name == "sm":
            return SLRM(**self.method_args)
        elif self.method_name in {"fib", "ent", "traj"}:
            return SCPD(sampling_type=self.method_name, **self.method_args)
        else:
            raise ValueError(f"Unknown method: {self.method_name}")

    def _get_method_args(self):
        method_cfg = getattr(self.cfg.method, self.method_name)
        return OmegaConf.to_container(method_cfg, resolve=True)

    def _prepare_all_metrics(self, trial_results, mc_true):
        self.mc_true = mc_true
        self.mc_ests = [r[0] for r in trial_results]
        self.diffs, self.costs = zip(*[(r[1], r[2]) for r in trial_results])
        self.errors = {"kld": {}, "frob": {}, "l1": {}}
        self.P_true, self.P_est = [], []
        self.Q_true, self.Q_est = [], []
        self.R_true, self.R_est = [], []

        for t in range(len(mc_true)):
            true_mc = mc_true[t]
            est_mc = self.mc_ests[t]

            P_true = true_mc.P
            P_est = est_mc.P
            Q_true = true_mc.Q
            Q_est = est_mc.Q
            R_true = true_mc.R
            R_est = est_mc.R

            if self.method_name in {"fib", "ent", "traj"}:
                D = P_true.ndim // 2
                Is = torch.tensor(P_true.shape[:D])
                I = torch.prod(Is).item()
                P_true = ten_to_mat(P_true, I)
                P_est = ten_to_mat(P_est, I)
                Q_true = ten_to_mat(Q_true, I)
                Q_est = ten_to_mat(Q_est, I)
                R_true = R_true.reshape(-1)
                R_est = R_est.reshape(-1)

            self.P_true.append(P_true)
            self.P_est.append(P_est)
            self.Q_true.append(Q_true)
            self.Q_est.append(Q_est)
            self.R_true.append(R_true)
            self.R_est.append(R_est)

            self.errors["kld"].setdefault("P", []).append(float(kld_err(P_true, P_est)))
            self.errors["kld"].setdefault("Q", []).append(float(kld_err(Q_true, Q_est)))
            self.errors["kld"].setdefault("R", []).append(float(kld_err(R_true, R_est)))

            self.errors["frob"].setdefault("P", []).append(
                float(normfrob_err(P_true, P_est))
            )
            self.errors["frob"].setdefault("Q", []).append(
                float(normfrob_err(Q_true, Q_est))
            )
            self.errors["frob"].setdefault("R", []).append(
                float(normfrob_err(R_true, R_est))
            )

            self.errors["l1"].setdefault("P", []).append(
                float(norml1_err(P_true, P_est))
            )
            self.errors["l1"].setdefault("Q", []).append(
                float(norml1_err(Q_true, Q_est))
            )
            self.errors["l1"].setdefault("R", []).append(
                float(norml1_err(R_true, R_est))
            )

    def _log_all(self):
        self._log_barplots()
        self._log_transition_matrices()
        if self.method_name != "sm":
            self._log_curveplots()

    def _log_barplots(self):
        for metric, values in self.errors.items():
            means = [np.mean(values[comp]) for comp in ["P", "Q", "R"]]
            labels = ["P", "Q", "R"]
            bar_data = [[val, lbl] for val, lbl in zip(means, labels)]

            table = wandb.Table(data=bar_data, columns=["value", "component"])
            wandb.log(
                {
                    f"{metric.upper()} Errors": wandb.plot.bar(
                        table, "component", "value", title=f"{metric.upper()} Errors"
                    )
                }
            )

    def _log_curveplots(self):
        min_len = min(len(x) for x in self.diffs)
        diffs_mean = np.mean([d[:min_len] for d in self.diffs], axis=0)
        costs_mean = np.mean([c[:min_len] for c in self.costs], axis=0)
        steps = list(range(min_len))

        diffs_plot = wandb.plot.line_series(
            xs=steps, ys=[diffs_mean], keys=[""], title="Mean Diffs", xname="Step"
        )
        costs_plot = wandb.plot.line_series(
            xs=steps, ys=[costs_mean], keys=[""], title="Mean Costs", xname="Step"
        )

        wandb.log({"Diffs (Mean)": diffs_plot, "Costs (Mean)": costs_plot})

    def _log_transition_matrices(self):
        trials = len(self.P_true)
        fig, axes = plt.subplots(nrows=2, ncols=trials, figsize=(2.5 * trials, 5))
        for t in range(trials):
            axes[0, t].imshow(self.P_true[t], cmap="viridis")
            axes[0, t].set_title(f"True P (trial {t})")
            axes[0, t].axis("off")

            axes[1, t].imshow(self.P_est[t], cmap="viridis")
            axes[1, t].set_title(f"Estimated P (trial {t})")
            axes[1, t].axis("off")

        plt.tight_layout()
        wandb.log({"Transition Matrices (per trial)": wandb.Image(fig)})
        plt.close(fig)

    def _save_results_as_json(self):
        os.makedirs(self.cfg.general.save_path, exist_ok=True)

        method_cfg = getattr(self.cfg.method, self.method_name)
        method_cfg_dict = OmegaConf.to_container(method_cfg, resolve=True)

        base_cfg = OmegaConf.to_container(self.cfg, resolve=True)
        base_cfg["method"] = {
            "method_name": self.method_name,
            self.method_name: method_cfg_dict,
        }

        matrices = {
            "P_true": [P.numpy().tolist() for P in self.P_true],
            "P_est": [P.numpy().tolist() for P in self.P_est],
            "Q_true": [Q.numpy().tolist() for Q in self.Q_true],
            "Q_est": [Q.numpy().tolist() for Q in self.Q_est],
            "R_true": [R.numpy().tolist() for R in self.R_true],
            "R_est": [R.numpy().tolist() for R in self.R_est],
        }

        result_dict = {
            "experiment": self.experiment_name,
            "config": base_cfg,
            "results": self.errors,
            "matrices": matrices,
        }

        path = os.path.join(self.cfg.general.save_path, f"{self.experiment_name}.json")
        with open(path, "w") as f:
            json.dump(result_dict, f, indent=4)
