import os

from omegaconf import OmegaConf
from easydict import EasyDict as edict

import torch

from src.utils import import_and_run_object


def parse_prefix(prefix, f, v):
    return f(*v[1:]) if type(v) is list and v[0] == prefix else v


def parse_config_item(v):
    v = parse_prefix('(env)', os.environ.get, v)
    v = parse_prefix('(obj)', import_and_run_object, v)

    return v


def parse_config(config):
    if isinstance(config, dict):
        return {k: parse_config(v) for k, v in config.items()}
    elif isinstance(config, list):
        return parse_config_item([parse_config(v) for v in config])
    else:
        return config


def process_config(config):
    acc = torch.accelerator.current_accelerator()
    config.setup.device = torch.device(f'{acc}:{config.setup.ddp.local_rank}')

    return config


def validate_config(config):
    assert config.model.d_model % config.model.n_heads == 0, "n_heads should divide d_model"


def load_config(path):
    config = OmegaConf.load(path)
    config = OmegaConf.to_container(config, resolve=True)
    config = parse_config(config)
    config = edict(config)
    
    config = process_config(config)
    validate_config(config)
    
    return config
