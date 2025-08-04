import math

import torch
import torch.nn as nn
import einx

from .transformer import Encoder


class DVSTDecoder(nn.Module):
    def __init__(self, config, pose_encoder):
        super().__init__()
        
        self.config = config
        self.C = self.config.C
        self.p = self.config.p
        self.d_model = self.config.d_model
        
        self.pose_encoder = pose_encoder
        
        self.transformer = Encoder(
            self.config.N_dec,
            self.config.d_model,
            self.config.n_heads,
            self.config.e_ff,
            self.config.qk_norm.enabled,
            self.config.qk_norm.eps,
            self.config.train.dropout,
            nn.GELU,
            self.config.attn_op
        )

        self.embeds_to_patch_embeds = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.C * self.p ** 2),
            nn.Sigmoid()
        )

    def forward(self, latent_embeds, video_query):
        q = video_query
        Kinv, R, t, time, hw = q.Kinv, q.R, q.t, q.time, q.shape[-2:]
        pose_embeds, pad = self.pose_encoder(Kinv, R, t, time, None, hw) # Computes query embeddings
        Is = []
        
        # For now, generating only a single image at a time
        # So we generate the images separately and then concatenate them after
        for embeds in pose_embeds:
            embeds = embeds.unsqueeze(0)

            embeds = torch.concat([latent_embeds, embeds], dim=-2) # Concats embeddings with latent embeddings
            embeds = self.transformer(embeds) # Creates image embeddings using transformer
            embeds = embeds[..., latent_embeds.shape[-2]:, :] # Discards embeddings mapped from latent embeddings
            embeds = self.embeds_to_patch_embeds(embeds) # Maps embeddings to patch embeddings

            I_padded = einx.rearrange('... (h w) (c p1 p2) -> ... c (h p1) (w p2)', embeds, h=math.isqrt(embeds.shape[-2]), c=self.C, p1=self.p, p2=self.p) # Maps patch embeddings back to image
            I = I_padded[pad[2]:I_padded.shape[-2]-pad[3], pad[0]:I_padded.shape[-1]-pad[1]]

            Is.append(I)
        
        return torch.concat(Is)
