import os

from omegaconf import OmegaConf
from easydict import EasyDict as edict

import torch

from src.utils import import_object


def process_config(config):
    # Torchrun already sets local and global ranks as environment variables
    config.setup.ddp.world_size = int(os.environ.get('WORLD_SIZE', 1))
    config.setup.ddp.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    config.setup.ddp.rank = int(os.environ.get('RANK', 0))
    config.setup.ddp.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    acc = torch.accelerator.current_accelerator()
    config.setup.device = torch.device(f'{acc}:{config.setup.ddp.local_rank}')
    
    config.setup.amp.dtype =import_object(config.setup.amp.dtype)

    config.model.attn_op = import_object(config.model.attn_op)
    config.model.latent_aggregator = import_object(config.model.latent_aggregator)
    
    return config


def validate_config(config):
    assert config.model.d_model % config.model.n_heads == 0, "n_heads should divide d_model"


def load_config(path):
    config = OmegaConf.load(path)
    config = OmegaConf.to_container(config, resolve=True)
    config = edict(config)
    
    config = process_config(config)
    validate_config(config)
    
    return config
