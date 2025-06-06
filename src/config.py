from dataclasses import dataclass, field
from typing import List, Optional

# TODO: Need to figure out how to give dimensions for tensor prediction

@dataclass
class GeneralConfig:
    seed: int = 1000
    save_path: str = "results/"
    use_wandb: bool = True


@dataclass
class ChainConfig:
    dims: List[int] = field(default_factory=lambda: [4, 4])
    data_path: str = "data/sim_matrix.npy"
    rank: int = 5
    burn_in: int = 500
    length: int = 300
    trials: int = 5


@dataclass
class DCConfig:
    K: int = 1
    alpha: float = 1e-1
    beta: float = 1.0
    gamma: float = 1.0
    pmin: float = 1e-6
    num_inn_itrs: int = 10
    inn_tol: float = 1e-7
    num_itrs: int = 2000
    tol: float = 1e-7
    verbose: bool = True


@dataclass
class NNConfig:
    K: Optional[int] = None
    beta: float = 1.0
    pmin: float = 1e-6
    gamma: float = 1.0
    verbose: bool = True
    num_itrs: int = 2000
    tol: float = 1e-7


@dataclass
class SMConfig:
    K: int = 5
    qmin: float = 1e-6


@dataclass
class SCPDConfig:
    K: int = 5
    qmin: float = 1e-6
    qmax: float = 1.0
    slide_window: int = 1000
    alpha_type: str = "adam"
    alpha_factor: float = 1.0
    alpha_weight: float = 1.0
    gamma_factor: float = 0.3
    gamma_weight: float = 0.3
    beta: float = 0.5
    eps: float = 1e-9
    B: int = 10
    B_max: int = 10
    increase_B: bool = False
    acceleration: bool = True
    num_itrs: int = 10000
    tol: float = 1e-7
    verbose: bool = True


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
    dataset: ChainConfig = field(default_factory=ChainConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
