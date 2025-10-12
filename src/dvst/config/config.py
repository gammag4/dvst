from dataclasses import dataclass, field
from typing import Callable
from torch import Tensor
from torch.nn import Module
from xformers.ops.fmha import AttentionFwOpBase, AttentionBwOpBase


from src.base.config import Config


@dataclass
class DVSTDatasetConfig:
    path: str
    scene_batch_size: int


@dataclass
class QKNormConfig:
    enabled: bool
    eps: float


@dataclass
class ModelTrainConfig:
    dropout: float


@dataclass
class DVSTModelConfig:
    p: int
    C: int
    n_oct: int

    N_enc: int
    N_dec: int
    d_model: int
    n_heads: int
    e_ff: int
    qk_norm: QKNormConfig
    attn_op: tuple[AttentionFwOpBase, AttentionBwOpBase]

    n_lat: int
    latent_aggregator: Callable[[Module, Tensor, Tensor], Tensor]

    train: ModelTrainConfig


@dataclass
class DVSTOptimizerConfig:
    lr: float
    betas: tuple[float, float]
    fused: bool


@dataclass
class DVSTLossSchedulerConfig:
    beta: int
    regime: str


@dataclass
class DVSTLossConfig:
    loss: str
    scheduler: DVSTLossSchedulerConfig


DVSTConfig = Config[DVSTDatasetConfig, DVSTModelConfig, DVSTOptimizerConfig, DVSTLossConfig]
