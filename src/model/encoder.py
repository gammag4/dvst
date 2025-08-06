import torch
import torch.nn as nn

from src.utils import create_bound_function

from .transformer import Encoder


class DVSTEncoder(nn.Module):
    # latent_aggregator(next_frame_embeds, current_latent_embeds) creates next latent embeds from current embeds and next frame
    def __init__(self, config, pose_encoder):
        super().__init__()
        
        self.config = config
        self.n_lat = self.config.n_lat
        self.d_model = self.config.d_model
        
        self.start_latent_embeds = nn.Parameter(torch.zeros((1, self.config.n_lat, self.config.d_model)))
        self.pose_encoder = pose_encoder
        self.latent_aggregator = create_bound_function(self, self.config.latent_aggregator)
        
        self.latent_norm = nn.LayerNorm(self.config.d_model) #TODO check RMSNorm
        self.transformer = Encoder(
            self.config.N_enc,
            self.config.d_model,
            self.config.n_heads,
            self.config.e_ff,
            self.config.qk_norm.enabled,
            self.config.qk_norm.eps,
            self.config.train.dropout,
            nn.GELU,
            self.config.attn_op
        )
        
    def forward(self, source_videos, latent_embeds=None):
        if latent_embeds is None:
            latent_embeds = self.latent_norm(self.start_latent_embeds)
        
        # Computes frame embeddings for all videos and gets each embedding
        pose_embeds = [f_embed for v in source_videos for f_embed in self.pose_encoder(v.Kinv, v.R, v.t, v.time, v.video)[0]]
            
        # Passes each frame embedding through the transformer aggregating into latent_embeds
        for embeds in pose_embeds:
            embeds = embeds.unsqueeze(0)

            latent_embeds = self.latent_aggregator(latent_embeds, embeds)

        return latent_embeds
