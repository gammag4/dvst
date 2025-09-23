import torch
import torch.nn as nn
import einx

from src.dvst.datasets.scene_dataset import View
from .transformer import Encoder
from .pose_encoder import PoseEncoder


class DVSTDecoder(nn.Module):
    def __init__(self, config, pose_encoder: PoseEncoder):
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
    
    def forward_batch(self, latent_embeds, query_view: View):
        Kinv, R, t, time, hw = query_view.Kinv, query_view.R, query_view.t, query_view.time, query_view.shape[-2:]
        pose_embeds, pad = self.pose_encoder(Kinv, R, t, time, None, hw) # Computes query embeddings
        h = (hw[0] + pad[2] + pad[3]) // self.p
        Is = []
        
        # For now, generating only a single image at a time
        # So we generate the images separately and then concatenate them after
        for embeds in pose_embeds:
            embeds = embeds.unsqueeze(0)

            embeds = torch.concat([latent_embeds, embeds], dim=-2) # Concats embeddings with latent embeddings
            final_embeds = self.transformer(embeds) # Creates image embeddings using transformer
            final_embeds = final_embeds[..., latent_embeds.shape[-2]:, :] # Discards embeddings mapped from latent embeddings
            final_embeds = self.embeds_to_patch_embeds(final_embeds) # Maps embeddings to patch embeddings

            I_padded = einx.rearrange('... (h w) (c p1 p2) -> ... c (h p1) (w p2)', final_embeds, h=h, c=self.C, p1=self.p, p2=self.p) # Maps patch embeddings back to image
            I = I_padded[..., pad[2]:I_padded.shape[-2]-pad[3], pad[0]:I_padded.shape[-1]-pad[1]]

            Is.append(I)
        
        return torch.concat(Is)
    
    def forward(self, latent_embeds, query_view: View):
        Is = []
        for batch in query_view:
            Is.append(self.forward_batch(latent_embeds, batch))
        
        return torch.concat(Is)
