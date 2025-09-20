from dataclasses import dataclass, field
from typing import Callable
from torch import Tensor
from xformers.ops.fmha import AttentionFwOpBase, AttentionBwOpBase, MemoryEfficientAttentionFlashAttentionOp

from src.base.config import create_config as create_config_base, DataloaderConfig, BaseDataConfig, BaseModelConfig, BaseOptimizerConfig

from src.dvst.model.encoder import DVSTEncoder
from src.dvst.model.latent_aggregators import residual_latent_aggregator
from src.dvst.model.loss import PerceptualLoss


@dataclass
class DVSTDataConfig(BaseDataConfig):
    path: str = 'res/tmp/'
    
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)


@dataclass
class QKNormConfig:
    # Enables QK-Norm
    enabled: bool = True
    # Epsilon for QK-Norm computation
    eps: float = 1e-4


@dataclass
class TrainConfig:
    # Dropout rate
    # TODO test separate dropouts for each layer
    dropout: float = 0.1
    loss: PerceptualLoss = PerceptualLoss()
    # loss: [(obj), torch.nn.MSELoss, null]


@dataclass
class DVSTModelConfig(BaseModelConfig):
    # Patch side length (the images will be broken into patches of size p x p)
    p: int = 16
    # Number of color channels in each frame
    C: int = 3
    # Number of octaves for representing each of the 6 components from plucker rays and the time component
    # If null, uses raw values instead
    # TODO change code to not use octaves/duplicate coordinates if n_oct = 0
    n_oct: int = 6

    # Number of layers in transformer encoder
    N_enc: int = 2
    # Number of layers in transformer decoder
    N_dec: int = 7
    # Dimension of all the vector representations used in the transformer models (dim for all vector inputs in transformers, not just the embedding vectors that will be the latent representations of scenes, idk why did i put such a confusing name but yeah)
    d_model: int = 192
    # Number of attention heads in both encoder and decoder (n_heads should divide d_model)
    n_heads: int = 12
    # Expansion factor for mlp blocks after attention blocks in each transformer block (embeddings will be expanded to e_ff * d_model dimensions then contracted back to d_model dimensions)
    e_ff: int = 4
    qk_norm: QKNormConfig = field(default_factory=QKNormConfig)
    # Operation used for attention, should be from xops.fmha
    attn_op: tuple[AttentionFwOpBase, AttentionBwOpBase] = MemoryEfficientAttentionFlashAttentionOp

    # Number of embedding vectors used as latent space representation for images
    # TODO could compute this based on frame size and video size
    n_lat: int = 256
    # The function that will be used to aggregate latents across frames. Should be one of the functions from src.model.latent_aggregators
    latent_aggregator: Callable[[DVSTEncoder, Tensor, Tensor], Tensor] = residual_latent_aggregator
    # TODO How many frames to break scenes into (use if using input scenes that are too big)
    # When using this, after the specified number of frames, the latent_embedding has its grad graph removed and becomes a leaf tensor
    #   Its gradients are not propagated back to the start_latent_embeds parameter, but this saves memory
    scene_batch_size: int = 6

    # frames_per_scene is the size of the batches that the videos will be broken into to create scenes
    #   TODO this was from the old idea of breaking scene into smaller batches of 3
    #   TODO test 3 ideas separate:
    #     just incremental creation of latents (testing both w or w/o residual)
    #     just breaking into blocks of 3
    #     and test both together (breaking into blocks of size k and doing incremental creation of latents)
    # frames_per_scene: 3

    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class DVSTOptimizerConfig(BaseOptimizerConfig):
    # Learning rate
    lr: float = 1e-4
    # AdamW betas
    betas: tuple[float, float] = (0.9, 0.95)
    # TODO Some places report issues so check if this gives errors or nans
    fused: bool = True


def create_config():
    return create_config_base(DVSTDataConfig(), DVSTModelConfig(), DVSTOptimizerConfig())
