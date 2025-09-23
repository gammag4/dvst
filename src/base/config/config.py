import os
from dataclasses import dataclass, field
from typing import Generic, TypeVar
from abc import ABC, abstractmethod
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
class AmpConfig:
    enabled: bool
    dtype: type


@dataclass
class GradScalerBatchRetryConfig:
    enabled: bool
    max_retries: int


@dataclass
class GradScalerConfig:
    enabled: bool
    batch_retry: GradScalerBatchRetryConfig


@dataclass
class GradManagerConfig:
    scaler: GradScalerConfig


# TODO maybe move out of config
@dataclass
class DistributedConfig:
    device: str = field(init=False)
    num_threads: int
    world_size: int
    local_world_size: int
    rank: int
    local_rank: int
    timeout: int
    
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
    benchmark_kernels: bool
    use_deterministic_algorithms: bool
    tf32_level: str
    amp: AmpConfig
    grad_manager: GradManagerConfig
    seed: int
    distributed: DistributedConfig


@dataclass
class GradClippingConfig:
    enabled: bool
    max_norm: float


TDatasetConfig = TypeVar('TDatasetConfig')


@dataclass
class DataloaderConfig:
    batch_size: int | None
    num_workers: int
    prefetch_factor: int | None
    shuffle: bool
    pin_memory: bool


@dataclass
class DataConfig(ABC, Generic[TDatasetConfig]):
    dataset: TDatasetConfig
    dataloader: DataloaderConfig


TOptimizerConfig = TypeVar('TOptimizerConfig')


@dataclass
class TrainConfig(Generic[TDatasetConfig, TOptimizerConfig]):
    data: DataConfig[TDatasetConfig]
    optimizer: TOptimizerConfig
    total_epochs: int
    save_every_passes: int
    checkpoints_folder_path: str
    grad_clipping: GradClippingConfig


TModelConfig = TypeVar('TModelConfig')


@dataclass
class Config(Generic[TDatasetConfig, TModelConfig, TOptimizerConfig]):
    model: TModelConfig
    train: TrainConfig[TDatasetConfig, TOptimizerConfig]
    setup: SetupConfig
