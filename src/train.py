import os
import wandb
import numpy as np
import torch
import json
from omegaconf import OmegaConf
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

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

        self.experiment_name = f"{self.method_name}_{self.method_args.get('K', 'NA')}"

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
            delayed(run_trial)(trajectories[t], mc_emp[t]) for t in range(self.cfg.general.trials)
        )
        self.trial_results = results
        self._log_all(mc_true)
        self.save_results_as_json(mc_true)

    def _instantiate_method(self):
        method = self.method_name
        args = self.method_args

        if method == "dc":
            return IPDC(**args)
        elif method == "nn":
            return SGSADMM(**args)
        elif method == "sm":
            return SLRM(**args)
        elif method in {"fib", "ent", "traj"}:
            return SCPD(sampling_type=method, **args)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _get_method_args(self):
        method_cfg = getattr(self.cfg.method, self.method_name)
        return OmegaConf.to_container(method_cfg, resolve=True)

    def _log_all(self, mc_true):
        trials = self.cfg.general.trials
        diffs, costs = zip(*[(r[1], r[2]) for r in self.trial_results])
        mc_ests = [r[0] for r in self.trial_results]

        # Final metrics
        klds = [kld_err(mc_true[t].P, mc_ests[t].P) for t in range(trials)]
        frobs = [normfrob_err(mc_true[t].P, mc_ests[t].P) for t in range(trials)]
        l1s = [norml1_err(mc_true[t].P, mc_ests[t].P) for t in range(trials)]

        fig, ax = plt.subplots()
        ax.boxplot([klds, frobs, l1s], labels=["KLD", "Frob", "L1"], patch_artist=True)
        ax.set_title("Error Distribution Across Trials")
        ax.set_ylabel("Error")
        wandb.log({"error_boxplot": wandb.Image(fig)})
        plt.close(fig)

        # Diffs and plots
        min_len = min(len(x) for x in diffs)
        diffs_mean = np.mean([d[:min_len] for d in diffs], axis=0)
        costs_mean = np.mean([c[:min_len] for c in costs], axis=0)
        steps = list(range(min_len))

        diffs_plot = wandb.plot.line_series(
            xs=steps,
            ys=[diffs_mean],
            keys=[""],
            title="Mean Diffs",
            xname="Step"
        )

        costs_plot = wandb.plot.line_series(
            xs=steps,
            ys=[costs_mean],
            keys=[""],
            title="Mean Costs",
            xname="Step"
        )

        wandb.log({
            "Diffs (Mean)": diffs_plot,
            "Costs (Mean)": costs_plot,
        })

        # Estimation
        fig, axes = plt.subplots(nrows=2, ncols=trials, figsize=(2.5 * trials, 5))
        for t in range(trials):
            ax_true = axes[0, t]
            ax_est = axes[1, t]
            ax_true.imshow(mc_true[t].P, cmap="viridis")
            ax_true.set_title(f"True P (trial {t})")
            ax_true.axis("off")
            ax_est.imshow(mc_ests[t].P, cmap="viridis")
            ax_est.set_title(f"Estimated P (trial {t})")
            ax_est.axis("off")

        plt.tight_layout()
        wandb.log({"Transition Matrices (per trial)": wandb.Image(fig)})
        plt.close(fig)

    def get_results(self):
        return self.trial_results

    def save_results_as_json(self, mc_true):
        os.makedirs(self.cfg.general.save_path, exist_ok=True)

        trials = self.cfg.general.trials
        mc_ests = [r[0] for r in self.trial_results]

        # Final metrics converted to floats
        klds = [float(kld_err(mc_true[t].P, mc_ests[t].P)) for t in range(trials)]
        frobs = [float(normfrob_err(mc_true[t].P, mc_ests[t].P)) for t in range(trials)]
        l1s = [float(norml1_err(mc_true[t].P, mc_ests[t].P)) for t in range(trials)]

        # Method-specific config only
        method_cfg = getattr(self.cfg.method, self.method_name)
        method_cfg_dict = OmegaConf.to_container(method_cfg, resolve=True)

        # Full config minus method subfields
        base_cfg = OmegaConf.to_container(self.cfg, resolve=True)
        base_cfg["method"] = {
            "method_name": self.method_name,
            self.method_name: method_cfg_dict,
        }

        result_dict = {
            "experiment": self.experiment_name,
            "config": base_cfg,
            "results": {
                "kld": klds,
                "frob": frobs,
                "l1": l1s,
            }
        }

        path = os.path.join(self.cfg.general.save_path, f"{self.experiment_name}.json")
        with open(path, "w") as f:
            json.dump(result_dict, f, indent=4)
