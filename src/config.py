import os

from omegaconf import OmegaConf
from easydict import EasyDict as edict

from src import latent_aggregators


def process_config(config):
    config.model.latent_aggregator = latent_aggregators.__dict__[config.model.latent_aggregator]
    
    # Torchrun already sets local and global ranks as environment variables
    config.setup.ddp.world_size = int(os.environ['WORLD_SIZE'])
    config.setup.ddp.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    config.setup.ddp.rank = int(os.environ['RANK'])
    config.setup.ddp.local_rank = int(os.environ['LOCAL_RANK'])
    
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
