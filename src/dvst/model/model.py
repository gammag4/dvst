from typing import Callable
import torch
import torch.nn as nn

from src.dvst.config import DVSTModelConfig
from src.dvst.datasets.scene_dataset import SourceBatch, QueryBatch
from .pose_encoder import PoseEncoder
from .encoder import DVSTEncoder
from .decoder import DVSTDecoder


# TODO change to RawDVST, create DVST that also has CNN to reduce dims and PoseWrapper to add a pose estimator to both
class DVST(nn.Module):
    # not specified: H, W, C, N_{context}
    # n_heads should divide d_model
    # p should divide H and W (padding, cropping and resizing)
    def __init__(self, config: DVSTModelConfig, loss: Callable | None = None):
        super().__init__()
        
        self.config = config
        
        self.pose_encoder = PoseEncoder(self.config)
        self.encoder = DVSTEncoder(self.config, self.pose_encoder)
        self.decoder = DVSTDecoder(self.config, self.pose_encoder)
        
        self.loss = loss # TODO check if eval mode or loss is none in forward
    
    @property
    def start_latent_embeds(self):
        return self.encoder.start_latent_embeds
    
    def create_scene_latents(self, batch: SourceBatch | list[SourceBatch], latent_embeds: torch.Tensor | list[torch.Tensor] | None | list[None]) -> torch.Tensor:
        latent_embeds = self.encoder(batch, latent_embeds)
        return latent_embeds
    
    def generate_frames(self, batch: QueryBatch | list[QueryBatch], latent_embeds: torch.Tensor | list[torch.Tensor] | None | list[None]) -> list[torch.Tensor] | torch.Tensor:
        return self.decoder(batch, latent_embeds)
    
    # Shape: (B, F, C, H, W)
    def forward(self, sources: SourceBatch | list[SourceBatch], targets: SourceBatch | list[SourceBatch], latent_embeds: torch.Tensor | list[torch.Tensor] | None | list[None]) -> tuple[torch.Tensor, torch.Tensor | list[torch.Tensor], dict]:
        latent_embeds = self.create_scene_latents(sources, latent_embeds)
        
        if isinstance(targets, list):
            queries = [QueryBatch(t.K, t.R, t.t, t.time, t.I.shape[-2:]) for t in targets]
            I2 = self.generate_frames(queries, latent_embeds)
            losses = [self.loss(t.I, res) for t, res in zip(targets, I2)]
            
            last_frames = {'gen': [i.detach() for i in I2], 'target': [t.I for t in targets]}
        else:
            queries = QueryBatch(targets.K, targets.R, targets.t, targets.time, targets.I.shape[-2:])
            I2 = self.generate_frames(queries, latent_embeds)
            losses = self.loss(targets.I, I2)
            
            last_frames = {'gen': [i for i in I2.detach()], 'target': [i for i in targets.I]}
        
        return losses, latent_embeds, last_frames
