from omegaconf import OmegaConf
from easydict import EasyDict as edict

from src import latent_aggregators


def process_config(config):
    config.model.latent_aggregator = latent_aggregators.__dict__[config.model.latent_aggregator]


def validate_config(config):
    assert config.model.d_model % config.model.n_heads == 0, "n_heads should divide d_model"


def load_config(path):
    config = OmegaConf.load(path)
    config = OmegaConf.to_container(config, resolve=True)
    config = edict(config)
    
    process_config(config)
    validate_config(config)
    
    return config
