from functools import partial
import torch
import torch.nn as nn
import einx

from src.dvst.config import DVSTModelConfig
from src.dvst.datasets.scene_dataset import SourceBatch
from .transformer import Encoder
from .pose_encoder import PoseEncoder


class DVSTEncoder(nn.Module):
    # latent_aggregator(next_frame_embeds, current_latent_embeds) creates next latent embeds from current embeds and next frame
    def __init__(self, config: DVSTModelConfig, pose_encoder: PoseEncoder):
        super().__init__()
        
        self.config = config
        
        self.pose_encoder = pose_encoder
        self.start_latent_embeds = nn.Parameter(torch.zeros((self.config.n_lat, self.config.d_model)))
        self.latent_aggregator = partial(self.config.latent_aggregator, self)
        self.latent_norm = nn.LayerNorm(self.config.d_model) #TODO check RMSNorm
        # self.latent_norm = lambda x: x # TODO
        
        self.transformer = Encoder(
            self.config.N_enc,
            self.config.d_model,
            self.config.d_attn,
            self.config.n_heads,
            self.config.e_ff,
            self.config.qk_norm.enabled,
            self.config.qk_norm.eps,
            self.config.train.dropout,
            nn.GELU,
            self.config.attn_op,
            self.config.use_activation_checkpointing
        )
    
    def _process_latents(self, latent_embeds: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        if latent_embeds is None:
            # (B, n_lat, d_model)
            latent_embeds = einx.rearrange('... -> b ...', self.latent_norm(self.start_latent_embeds), b=batch_size)
        
        return latent_embeds
    
    def _forward_tensor(self, batch: SourceBatch, latent_embeds: torch.Tensor | None) -> torch.Tensor:
        # K: (B, F, 3, 3), R: (B, F, 3, 3), t: (B, F, 3), time: (B, F), I: (B, F, C, H, W)
        K, R, t, time, I = batch.K, batch.R, batch.t, batch.time, batch.I
        
        latent_embeds = self._process_latents(latent_embeds, I.shape[0])
        
        # Computes frame embeddings for all videos and gets each embedding
        pose_embeds, _ = self.pose_encoder(K=K, R=R, t=t, time=time, I=I) # (B, F, n_lat, d_model)
        pose_embeds = einx.rearrange('b f ... -> f b ...', pose_embeds) # (F, B, ...)
        
        # TODO use something like RAG to choose which embeds to use from which frames/views and which order to use them in the context window
        # Passes each frame embedding through the transformer aggregating into latent_embeds
        for frame_embeds in pose_embeds:
            latent_embeds = self.latent_aggregator(latent_embeds, frame_embeds)
            # for i in range(frame_embeds.shape[0]):
            #     latent_embeds[i] = self.latent_aggregator(latent_embeds[i], frame_embeds[i])
        
        return latent_embeds
    
    # When the batch is a list instead of tensor
    def _forward_list(self, batch: list[SourceBatch], latent_embeds: list[torch.Tensor] | list[None]) -> list[torch.Tensor]:
        ls = []
        for b, l in zip(batch, latent_embeds):
            b.K, b.R, b.t, b.time, b.I, l = [k for k in (b.K, b.R, b.t, b.time, b.I, l)]
            ls.append(self._forward_tensor(b, l))
        
        return ls
    
    def forward(self, batch: SourceBatch | list[SourceBatch], latent_embeds: torch.Tensor | list[torch.Tensor] | None | list[None]) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(batch, list) and isinstance(latent_embeds, list):
            return self._forward_list(batch, latent_embeds)
        if isinstance(batch, torch.Tensor) and isinstance(latent_embeds, torch.Tensor):
            return self._forward_tensor(batch, latent_embeds)
        
        raise Exception(f'Wrong types for batch and latent_embeds: {(type(batch), type(latent_embeds))}')
