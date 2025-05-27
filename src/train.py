import os
import wandb
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from joblib import Parallel, delayed

from src.utils import kld_err, normfrob_err, norml1_err, ten_to_mat
from src.estimation.baselines import IPDC, SGSADMM, SLRM
from src.estimation.scpd import SCPD


class Trainer:
    def __init__(self, experiment_name: str, cfg, I: int):
        self.cfg = cfg
        self.I = I
        self.experiment_name = experiment_name
        self.method_name = cfg.method.method_name

        self.method_args = self._get_method_args()
        self.method_instance = self._instantiate_method()
        self.trial_results = []

        if self.cfg.general.use_wandb:
            wandb.init(
                project="markov-chain-estimation",
                config=OmegaConf.to_container(cfg, resolve=True),
                name=self.experiment_name,
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
            delayed(run_trial)(trajectories[t], mc_emp[t])
            for t in range(self.cfg.general.trials)
        )
        self.trial_results = results
        self._log_all(mc_true)
        self._save_results_as_json(mc_true)

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

    def _log_all(self, mc_true):
        _ = self.cfg.general.trials
        diffs, costs = zip(*[(r[1], r[2]) for r in self.trial_results])
        mc_ests = [r[0] for r in self.trial_results]

        errors = self._compute_all_errors(mc_true, mc_ests)
        self._log_barplots(errors)
        self._log_transition_matrices(mc_true, mc_ests)
        if self.method_name != "sm":
            self._log_curveplots(diffs, costs)

    def _compute_all_errors(self, mc_true, mc_ests):
        errors = {"kld": {}, "frob": {}, "l1": {}}

        for attr in ["P", "Q", "R"]:
            true_list = [getattr(mc, attr) for mc in mc_true]
            est_list = [getattr(mc, attr) for mc in mc_ests]

            errors["kld"][attr] = [
                float(kld_err(true_list[t], est_list[t])) for t in range(len(mc_true))
            ]
            errors["frob"][attr] = [
                float(normfrob_err(true_list[t], est_list[t]))
                for t in range(len(mc_true))
            ]
            errors["l1"][attr] = [
                float(norml1_err(true_list[t], est_list[t]))
                for t in range(len(mc_true))
            ]

        return errors

    def _log_barplots(self, errors):
        for metric, values in errors.items():
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

    def _log_curveplots(self, diffs, costs):
        min_len = min(len(x) for x in diffs)
        diffs_mean = np.mean([d[:min_len] for d in diffs], axis=0)
        costs_mean = np.mean([c[:min_len] for c in costs], axis=0)
        steps = list(range(min_len))

        diffs_plot = wandb.plot.line_series(
            xs=steps, ys=[diffs_mean], keys=[""], title="Mean Diffs", xname="Step"
        )
        costs_plot = wandb.plot.line_series(
            xs=steps, ys=[costs_mean], keys=[""], title="Mean Costs", xname="Step"
        )

        wandb.log({"Diffs (Mean)": diffs_plot, "Costs (Mean)": costs_plot})

    def _log_transition_matrices(self, mc_true, mc_ests):
        trials = self.cfg.general.trials
        fig, axes = plt.subplots(nrows=2, ncols=trials, figsize=(2.5 * trials, 5))
        for t in range(trials):
            P_true = mc_true[t].P
            P_est = mc_ests[t].P
            if self.method_name in ["fib", "ent", "traj"]:
                D = P_true.ndim // 2
                Is = torch.tensor(P_true.shape[:D])
                I = torch.prod(Is).item()
                P_true = ten_to_mat(P_true, I)
                P_est = ten_to_mat(P_est, I)
            axes[0, t].imshow(P_true, cmap="viridis")
            axes[0, t].set_title(f"True P (trial {t})")
            axes[0, t].axis("off")
            axes[1, t].imshow(P_est, cmap="viridis")
            axes[1, t].set_title(f"Estimated P (trial {t})")
            axes[1, t].axis("off")

        plt.tight_layout()
        wandb.log({"Transition Matrices (per trial)": wandb.Image(fig)})
        plt.close(fig)

    def _save_results_as_json(self, mc_true):
        os.makedirs(self.cfg.general.save_path, exist_ok=True)
        mc_ests = [r[0] for r in self.trial_results]
        errors = self._compute_all_errors(mc_true, mc_ests)

        method_cfg = getattr(self.cfg.method, self.method_name)
        method_cfg_dict = OmegaConf.to_container(method_cfg, resolve=True)

        base_cfg = OmegaConf.to_container(self.cfg, resolve=True)
        base_cfg["method"] = {
            "method_name": self.method_name,
            self.method_name: method_cfg_dict,
        }

        result_dict = {
            "experiment": self.experiment_name,
            "config": base_cfg,
            "results": errors,
        }

        path = os.path.join(self.cfg.general.save_path, f"{self.experiment_name}.json")
        with open(path, "w") as f:
            json.dump(result_dict, f, indent=4)
