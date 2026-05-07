import torch
from xformers.ops.fmha import MemoryEfficientAttentionFlashAttentionOp

from src.base.config import *
from src.base.providers import ConfigProvider

from .config import *
from src.dvst.model.latent_aggregators import residual_latent_aggregator


class DVSTConfigProvider(ConfigProvider[DVSTDatasetConfig, DVSTModelConfig, DVSTOptimizerConfig, DVSTLossConfig]):
    def __init__(self):
        super().__init__()
    
    def _create_default_config(self) -> DVSTConfig:
        return Config(
            # Configs for setting up train/eval environments
            setup=SetupConfig(
                # Chooses whether to benchmark kernels
                # TODO Disabled for now since the input data will have variable sizes
                # TODO check if it is rerunning benchmarks
                benchmark_kernels=False,
                # Uses only deterministic algorithms, impacts performance a lot
                use_deterministic_algorithms=False,
                # Sets precision for float32 operations in CUDA matmul and cuDNN convolutions
                # 'highest' disables tf32
                # 'high' and 'medium' enable tf32
                tf32_level='high',
                # Automatic mixed precision config
                amp=AmpConfig(
                    # Enables/disables AMP
                    enabled=True,
                    # Sets AMP data type
                    dtype=torch.bfloat16
                ),
                # Gradient manager config
                grad_manager=GradManagerConfig(
                    # Gradient scaler config
                    scaler=GradScalerConfig(
                        # Enables/disables grad scaling
                        enabled=False,
                        # Batch retry config
                        batch_retry=GradScalerBatchRetryConfig(
                            # Enables/disables batch retrying
                            enabled=False,
                            # Sets maximum number of retries for a batch before giving up
                            max_retries=4
                        )
                    )
                ),
                # Seed for computations
                seed=42,
                # Distributed config
                distributed=DistributedConfig(
                    # Device used in current process. It is automatically computed
                    #device=None,
                    # Number of threads running in each process. If None, it is computed from cpu count and topology
                    num_threads=None,
                    # Number of process in the process group. If None, tries to get from environment variable set by torchrun and, if fails, defaults to 1
                    world_size=None,
                    # Number of processes locally. If None, tries to get from environment variable set by torchrun and, if fails, defaults to 1
                    local_world_size=None,
                    # Id of current process globally. If None, tries to get from environment variable set by torchrun and, if fails, defaults to 0
                    rank=None,
                    # Id of current process locally. If None, tries to get from environment variable set by torchrun and, if fails, defaults to 0
                    local_rank=None,
                    # Timeout for distributed process group
                    timeout=1800
                )
            ),
            train=TrainConfig(
                # Number of scenes in each batch
                batch_size=8,
                # How many train steps to run in each epoch
                n_train_steps=200000,
                # Save after every n passes (forward/backward pass)
                save_every_steps=1000,
                log_every_passes=10,
                checkpoints=CheckpointsConfig(
                    # Folder in which to save checkpoints
                    folder_path='res/tmp/checkpoint/',
                    # Whether to load only weights, should be false only for checkpoints from trusted sources
                    weights_only=False
                ),
                # Gradient clipping config
                grad_clipping=GradClippingConfig(
                    # Whether to use gradient clipping
                    enabled=True,
                    # Maximum gradient norm for gradient clipping
                    max_norm=1.0
                ),
                data=DataConfig(
                    dataloader=DataloaderConfig(
                        batch_size=None,
                        num_workers=4, # TODO 4 per gpu, needs to be checked (source: https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5)
                        prefetch_factor=None,
                        shuffle=True,
                        pin_memory=True # TODO check
                    ),
                    dataset=DVSTDatasetConfig(
                        # Path for datasets
                        path='res/tmp/',
                    )
                ),
                optimizer=DVSTOptimizerConfig(
                    # Number of warmup steps before going to cosine learning rate
                    n_warmup_steps=15000,
                    # Learning rate
                    lr=4e-4,
                    # AdamW betas
                    betas=(0.999, 0.9999),
                    weight_decay=0.01,
                    # TODO Some places report issues so check if this gives errors or nans
                    fused=True
                ),
                loss=DVSTLossConfig(
                    # Loss to use, can be either 'mse' or 'perceptual'
                    loss='perceptual',
                    scheduler=DVSTLossSchedulerConfig(
                        # Beta for perceptual loss scheduler
                        beta=0.2,
                        # Regime to use for perceptual loss scheduler, can be either 'constant', 'deep_to_shallow' or 'shallow_to_deep'
                        regime='constant',
                        # regime='deep_to_shallow',
                    )
                )
            ),
            model=DVSTModelConfig(
                # Patch side length (the images will be broken into patches of size p x p)
                p=8,
                # Number of color channels in each frame
                C=3,
                # Number of octaves for representing each of the 6 components from plucker rays and the time component
                # If None, uses raw values instead
                n_oct=None,
                # Whether to use plucker rays or raw origin-direction vector pairs
                use_plucker=True,
                
                # Number of layers in transformer encoder
                N_enc=12,
                # Number of layers in transformer decoder
                N_dec=12,
                # Dimension of all the vector representations used in the transformer models except attention (dim for all vector inputs in transformers, not just the embedding vectors that will be the latent representations of scenes, idk why did i put such a confusing name but yeah)
                d_model=384,
                # Dimension of vector representations in the attention space of transformer block (normally chosen to be equal to d_model)
                d_attn=384,
                # Number of attention heads in both encoder and decoder (n_heads should divide d_model)
                n_heads=12,
                # Expansion factor for mlp blocks after attention blocks in each transformer block (embeddings will be expanded to e_ff * d_model dimensions then contracted back to d_model dimensions)
                e_ff=4,
                # Query-Key Normalization config
                qk_norm=QKNormConfig(
                    # Enables QK-Norm
                    enabled=True,
                    # Epsilon for QK-Norm computation
                    eps=1e-12
                ),
                # Operation used for attention, should be from xops.fmha
                attn_op=MemoryEfficientAttentionFlashAttentionOp,
                use_activation_checkpointing=True,
                
                # Number of embedding vectors used as latent space representation for images
                # TODO could compute this based on frame size and video size
                n_lat=56, # 2 images
                # The function that will be used to aggregate latents across frames. Should be one of the functions from src.model.latent_aggregators
                latent_aggregator=residual_latent_aggregator,
                
                # frames_per_scene is the size of the batches that the videos will be broken into to create scenes
                #   TODO this was from the old idea of breaking scene into smaller batches of 3
                #   TODO test 3 ideas separate:
                #     just incremental creation of latents (testing both w or w/o residual)
                #     just breaking into blocks of 3
                #     and test both together (breaking into blocks of size k and doing incremental creation of latents)
                # frames_per_scene: 3
                
                train=ModelTrainConfig(
                    # Dropout rate
                    # TODO test separate dropouts for each layer
                    dropout=0.1
                )
            )
        )
