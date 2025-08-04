import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict as edict

from src.utils import get_videos_slice

from .pose_encoder import PoseEncoder
from .encoder import DVSTEncoder
from .decoder import DVSTDecoder

# TODO change to RawDVST, create DVST that also has CNN to reduce dims and PoseWrapper to add a pose estimator to both
class DVST(nn.Module):
    # not specified: H, W, C, N_{context}
    # n_heads should divide d_model
    # p should divide H and W (padding, cropping and resizing)
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.scene_batch_size = self.config.scene_batch_size
        
        self.start_latent_embeds = nn.Parameter(torch.zeros((1, self.config.n_lat, self.config.d_model)))
        self.pose_encoder = PoseEncoder(self.config)
        self.encoder = DVSTEncoder(self.config, self.pose_encoder)
        self.decoder = DVSTDecoder(self.config, self.pose_encoder)
        
        # TODO
        self.loss_fn = F.mse_loss

    def create_scene_latents(self, videos, n_frames):
        latent_embeds = self.start_latent_embeds
        
        for i in range(0, n_frames):
            curr_videos = get_videos_slice(videos, i, i + 1)
            latent_embeds = self.encoder(latent_embeds, curr_videos)
            
        return latent_embeds

    def generate_frames(self, latent_embeds, video_query):
        return self.decoder(latent_embeds, video_query)
    
    # We assume videos are not big enough so that they need to be loaded in batches into memory #TODO load in batches if size exceed n_frames (create new scene for each batch of n_frames)
    def forward(self, videos, queries, targets, start, end, latent_embeds):
        loss = 0.
        
        curr_videos = get_videos_slice(videos, start, end)
        curr_queries = get_videos_slice(queries, start, end)
        curr_targets = get_videos_slice(targets, start, end)
        
        latent_embeds = self.encoder(latent_embeds, curr_videos)
        
        for query, target in zip(curr_queries, curr_targets):
            frames = self.generate_frames(latent_embeds, query)
            l = self.loss_fn(frames, target) / frames[0].numel()
            loss = loss + l
        
        return loss, latent_embeds
