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
        self.use_incremental_aggregator = self.config.use_incremental_aggregator
        
        # TODO move embeds to model and use as input in forward (similar to decoder)
        self.start_latent_embeds = nn.Parameter(torch.zeros((1, self.n_lat, self.d_model)))
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
        
    def forward(self, videos, n_frames):
        # Computes all frame embeddings and aggregates them at once
        if not self.use_incremental_aggregator:
            scene_embeds = []
            for v in videos:
                Kinv = v.Kinv
                R = v.R if v.R.shape[0] == 1 else v.R[:n_frames]
                t = v.t if v.t.shape[0] == 1 else v.t[:n_frames]
                time = v.time[:n_frames]
                I = v.video[:n_frames]
                
                video_embeds, _ = self.pose_encoder(Kinv, R, t, time, I) # Computes embeddings for entire video
                scene_embeds.append(video_embeds)
            embeds = self.latent_aggregator(self.start_latent_embeds, scene_embeds) # Aggregates embeddings for entire scene
                    
            return embeds
            
        # For each frame time, gets all frames from all cams in that time, passes them through the transformer and then goes to the next frame time
        current_embeds = self.start_latent_embeds
        for i in range(n_frames):
            for v in videos:
                Kinv = v.Kinv
                R = v.R if v.R.shape[0] == 1 else v.R[i:i+1]
                t = v.t if v.t.shape[0] == 1 else v.t[i:i+1]
                time = v.time[i:i+1]
                I = v.video[i:i+1]
                
                frame_embeds, _ = self.pose_encoder(Kinv, R, t, time, I) # Computes frame embeddings
                current_embeds = self.latent_aggregator(frame_embeds, current_embeds) # Aggregats with previous embeddings
                
        return current_embeds
