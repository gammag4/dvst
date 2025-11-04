import torch
import torch.nn as nn
import einx

from src.dvst.config import DVSTModelConfig
from src.dvst.datasets.scene_dataset import QueryBatch
from .transformer import Encoder
from .pose_encoder import PoseEncoder


class DVSTDecoder(nn.Module):
    def __init__(self, config: DVSTModelConfig):
        super().__init__()
        
        self.config = config
        self.C = self.config.C
        self.p = self.config.p
        
        self.pose_encoder = PoseEncoder(is_decoder=True, config=self.config)
        self.latent_norm = nn.LayerNorm(self.config.d_model)
        
        self.transformer = Encoder(
            self.config.N_dec,
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
        
        self.embeds_to_patch_embeds = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Linear(in_features=self.config.d_model, out_features=self.C * self.p ** 2),
            nn.Sigmoid()
        )
    
    def _forward_tensor(self, batch: QueryBatch, latent_embeds: torch.Tensor | None) -> torch.Tensor:
        # K: (B, F, 3, 3), R: (B, F, 3, 3), t: (B, F, 3), time: (B, F), hw: (2,)
        K, R, t, time, hw = batch.K, batch.R, batch.t, batch.time, batch.hw
        
        pose_embeds, pad = self.pose_encoder(K=K, R=R, t=t, time=time, hw=hw) # Computes query embeddings, (B, F, n_lat, d_model)
        pose_embeds = einx.rearrange('b f ... -> f b ...', pose_embeds) # (F, B, ...)
        
        latent_embeds = einx.rearrange('... -> f ...', latent_embeds, f=pose_embeds.shape[0])
        
        h = (hw[0] + pad[2] + pad[3]) // self.p
        
        embeds = pose_embeds
        embeds = torch.concat([latent_embeds, embeds], dim=-2) # Concats embeddings with latent embeddings
        embeds = self.latent_norm(embeds) # Transforms to latent space
        final_embeds = self.transformer(embeds) # Creates image embeddings using transformer
        final_embeds = final_embeds[..., latent_embeds.shape[-2]:, :] # Discards embeddings mapped from latent embeddings
        final_embeds = self.embeds_to_patch_embeds(final_embeds) # Maps embeddings to patch embeddings
        
        I_padded = einx.rearrange('... (h w) (c p1 p2) -> ... c (h p1) (w p2)', final_embeds, h=h, c=self.C, p1=self.p, p2=self.p) # Maps patch embeddings back to image
        I = I_padded[..., pad[2]:I_padded.shape[-2]-pad[3], pad[0]:I_padded.shape[-1]-pad[1]]
        I = einx.rearrange('f b ... -> b f ...', I) # (B, F, ...)
        
        return I
    
    # When the batch is a list instead of tensor
    def _forward_list(self, batch: list[QueryBatch], latent_embeds: torch.Tensor | list[torch.Tensor] | None | list[None]) -> list[torch.Tensor]:
        Is = []
        for b, l in zip(batch, latent_embeds):
            b.K, b.R, b.t, b.time, b.hw = b.K, b.R, b.t, b.time, b.hw
            Is.append(self._forward_tensor(b, l))
        
        return Is
    
    def forward(self, batch: QueryBatch | list[QueryBatch], latent_embeds: torch.Tensor | None = None) -> list[torch.Tensor] | torch.Tensor:
        if isinstance(batch, list):
            return self._forward_list(batch, latent_embeds)
        
        return self._forward_tensor(batch, latent_embeds)
