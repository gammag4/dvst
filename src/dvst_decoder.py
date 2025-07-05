import torch
import torch.nn as nn
import einx


class DVSTDecoder(nn.Module):
    def __init__(self, dvst_config, pose_encoder):
        super().__init__()
        
        self.dvst_config = dvst_config
        self.C = self.dvst_config.C
        self.p = self.dvst_config.p
        self.d_lat = self.dvst_config.d_lat
        
        self.pose_encoder = pose_encoder
        self.transformer = lambda x: x #TODO
        self.embeds_to_patch_embeds = nn.Sequential([
            nn.Linear(in_features=self.d_lat, out_features=self.C * self.p ** 2),
            nn.Sigmoid()
        ])

    # For now, generating only a single image at a time is supported, so, in all tensors, B=1
    def forward(self, latent_embeds, Kinv, R, t, time, hw):
        embeds, pad = self.pose_encoder(Kinv, R, t, time, None, hw) # Computes query embeddings
        embeds = torch.concat([latent_embeds, embeds], dim=-2) # Concats embeddings with latent embeddings
        embeds = self.transformer(embeds) # Creates image embeddings using transformer
        embeds = embeds[latent_embeds.shape[-2]:] # Discards embeddings mapped from latent embeddings
        embeds = self.embeds_to_patch_embeds(embeds) # Maps embeddings to patch embeddings
        I_padded = einx.rearrange('... (h w) (c p1 p2) -> ... c (h p1) (w p2)', embeds, c=self.C, p1=self.p, p2=self.p) # Maps patch embeddings back to image
        I = I_padded[pad[2]:I_padded.shape[-2]-pad[3], pad[0]:I_padded.shape[-1]-pad[1]]
        
        return I
