from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class GeneralConfig:
    seed: int = 1000
    save_path: str = "results/"
    data_path: str = "data/"
    trials: int = 5
    use_wandb: bool = True

@dataclass
class ChainConfig:
    dims: List[int] = field(default_factory=lambda: [4, 4])
    rank: int = 5
    burn_in: int = 500
    length: int = 300

@dataclass
class TrainingConfig:
    num_iters: int = 1000
    tol: float = 1e-7
    slide_window: int = 1000
    eval_every: int = 50
    save: bool = True

@dataclass
class DCConfig:  # renamed from IPDCConfig
    alpha: float = 1e-1
    beta: float = 1.0
    gamma: float = 1.0
    pmin: float = 1e-6
    num_inn_itrs: int = 10
    inn_tol: float = 1e-7
    verbose: bool = True
    K: int = 5

@dataclass
class NNConfig:  # renamed from SGSADMMConfig
    beta: float = 1.0
    pmin: float = 1e-6
    gamma: float = 1.0
    verbose: bool = True

@dataclass
class SMConfig:  # renamed from SLRMConfig
    qmin: float = 1e-6

@dataclass
class SCPDConfig:
    qmin: float = 1e-6
    qmax: float = 1.0
    tol: float = 1e-8
    slide_window: int = 1000
    ALPHA_TYPE: str = "adam"
    alpha_factor: float = 1.0
    alpha_weight: float = 1.0
    gamma_factor: float = 0.3
    gamma_weight: float = 0.3
    beta: float = 0.5
    eps: float = 1e-9
    B: Optional[int] = None
    INCREASE_B: bool = False
    ACCELERATION: bool = True

@dataclass
class MethodConfig:
    method_name: str = "dc"

    dc: DCConfig = field(default_factory=DCConfig)
    nn: NNConfig = field(default_factory=NNConfig)
    sm: SMConfig = field(default_factory=SMConfig)
    fib: SCPDConfig = field(default_factory=SCPDConfig)
    ent: SCPDConfig = field(default_factory=SCPDConfig)
    traj: SCPDConfig = field(default_factory=SCPDConfig)

@dataclass
class MainConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    chain: ChainConfig = field(default_factory=ChainConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
