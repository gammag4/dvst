import os
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Generic
import torch


def parse_env(name, default):
    res = os.environ.get(name, default)
    try:
        res = float(res)
        res = int(res)
    except ValueError:
        pass
    
    return res


def parse_env_default(var_name, current_value, default_value):
    return parse_env(var_name, default_value) if current_value is None else current_value


@dataclass
class DataloaderConfig:
    batch_size: int | None = None
    num_workers: int = 4 # TODO 4 per gpu, needs to be checked (source: https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5)
    prefetch_factor: int | None = None
    shuffle: bool = True
    pin_memory: bool = True # TODO check


@dataclass
class AmpConfig:
    # Enables/disables AMP
    enabled: bool = True
    # Sets AMP data type
    dtype: type = torch.bfloat16


@dataclass
class GradScalerBatchRetryConfig:
    # Enables/disables batch retrying
    enabled: bool = False
    # Sets maximum number of retries for a batch before giving up
    max_retries: int = 4


@dataclass
class GradScalerConfig:
    # Enables/disables grad scaling
    enabled: bool = False
    # Batch retry config
    batch_retry: GradScalerBatchRetryConfig = field(default_factory=GradScalerBatchRetryConfig)


@dataclass
class GradManagerConfig:
    # Gradient scaler config
    scaler: GradScalerConfig = field(default_factory=GradScalerConfig)


# TODO maybe move out of config
@dataclass
class DistributedConfig:
    # Device used in current process. It is automatically computed and should be null
    device: str = field(init=False)
    # Number of threads running in each process. If None, it is computed from cpu count and topology
    num_threads: int = None
    # Number of process in the process group. If None, tries to get from environment variable set by torchrun and, if fails, defaults to 1
    world_size: int = None
    # Number of processes locally. If None, tries to get from environment variable set by torchrun and, if fails, defaults to 1
    local_world_size: int = None
    # Id of current process globally. If None, tries to get from environment variable set by torchrun and, if fails, defaults to 0
    rank: int = None
    # Id of current process locally. If None, tries to get from environment variable set by torchrun and, if fails, defaults to 0
    local_rank: int = None
    # Timeout for ddp process group
    timeout: int = 1800
    
    def __post_init__(self):
        self.world_size = parse_env_default('WORLD_SIZE', self.world_size, 1)
        self.local_world_size = parse_env_default('LOCAL_WORLD_SIZE', self.local_world_size, 1)
        self.rank = parse_env_default('RANK', self.rank, 0)
        self.local_rank = parse_env_default('LOCAL_RANK', self.local_rank, 0)
        
        if self.num_threads is None:
            num_cpus = os.cpu_count()
            self.num_threads = num_cpus // self.local_world_size + (1 if self.local_rank > num_cpus % self.local_world_size else 0)

        acc = torch.accelerator.current_accelerator()
        self.device = f'{acc}:{self.local_rank}'


@dataclass
class SetupConfig:
    # Chooses whether to benchmark kernels
    # TODO Disabled for now since the input data will have variable sizes
    # TODO check if it is rerunning benchmarks
    benchmark_kernels: bool = False
    # Uses only deterministic algorithms, impacts performance a lot
    use_deterministic_algorithms: bool = False
    # Sets precision for float32 operations in CUDA matmul and cuDNN convolutions
    # 'highest' disables tf32
    # 'high' and 'medium' enable tf32
    tf32_level: str = 'high'
    # Automatic mixed precision config
    amp: AmpConfig = field(default_factory=AmpConfig)
    # Gradient manager config
    grad_manager: GradManagerConfig = field(default_factory=GradManagerConfig)
    # Seed for computations
    seed: int = 42
    # Distributed config
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


@dataclass
class GradClippingConfig:
    # Whether to use gradient clipping
    enabled: bool = True
    # Maximum gradient norm for gradient clipping
    max_norm: float = 1.0


@dataclass
class BaseDataConfig(ABC):
    pass


@dataclass
class BaseOptimizerConfig(ABC):
    pass


@dataclass
class TrainConfig:
    data: BaseDataConfig
    optimizer: BaseOptimizerConfig
    total_epochs: int = 1
    save_every_passes: int = 100
    checkpoints_folder_path: str = 'res/tmp/checkpoint/'
    grad_clipping: GradClippingConfig = field(default_factory=GradClippingConfig)


@dataclass
class BaseModelConfig(ABC):
    pass


@dataclass
class Config:
    model: BaseModelConfig
    train: TrainConfig
    # Configs used to set up train/eval environments
    setup: SetupConfig = field(default_factory=SetupConfig)


def create_config(data: BaseDataConfig, model: BaseModelConfig, optimizer: BaseOptimizerConfig):
    return Config(model=model, train=TrainConfig(data=data, optimizer=optimizer))
