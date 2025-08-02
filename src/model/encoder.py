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
        
        self.start_latent_embeds = nn.Parameter(torch.zeros((self.n_lat, self.d_model)))
        self.pose_encoder = pose_encoder
        self.latent_aggregator = create_bound_function(self, self.config.latent_aggregator)
        
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
        
    def forward(self, scene):
        # For each frame time, gets all frames from all cams in that time, passes them through the transformer and then goes to the next frame time

        current_embeds = self.start_latent_embeds
        for i in range(scene.n_frames):
            for s in scene.videos:
                Kinv, R, t, time, video = [s[i] for i in ('Kinv', 'R', 't', 'time', 'video')]
                I = video[i]
                
                frame_embeds, _ = self.pose_encoder(Kinv, R, t, time, I) # Computes frame embeddings
                current_embeds = self.latent_aggregator(frame_embeds, current_embeds)
                
        return current_embeds
