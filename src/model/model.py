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
        
        self.pose_encoder = PoseEncoder(self.config)
        self.encoder = DVSTEncoder(self.config, self.pose_encoder)
        self.decoder = DVSTDecoder(self.config, self.pose_encoder)
        self.start_latent_embeds = self.encoder.start_latent_embeds
        
        self.loss = config.train.loss
        
    def get_start_latent_embeds(self):
        return self.encoder.start_latent_embeds

    def create_scene_latents(self, videos, n_frames):
        latent_embeds = None
        
        for i in range(0, n_frames):
            curr_videos = get_videos_slice(videos, i, i + 1)
            latent_embeds = self.encoder(curr_videos, latent_embeds)
            
        return latent_embeds

    def generate_frames(self, latent_embeds, video_query):
        return self.decoder(latent_embeds, video_query)
    
    # We assume videos are not big enough so that they need to be loaded in batches into memory #TODO load in batches if size exceed n_frames (create new scene for each batch of n_frames)
    def forward(self, videos, queries, targets, start, end, latent_embeds=None):
        loss = 0.
        
        curr_videos = get_videos_slice(videos, start, end)
        curr_queries = get_videos_slice(queries, start, end) # TODO fix so that it starts from the beginning
        curr_targets = get_videos_slice(targets, start, end)
        
        latent_embeds = self.encoder(curr_videos, latent_embeds)
        
        for query, target in zip(curr_queries, curr_targets):
            frames = self.generate_frames(latent_embeds, query)
            l = self.loss(frames, target.video) # TODO fix so that frames from starting batches receive less weights every time since they get repeated more
            loss = loss + l
        
        return loss, latent_embeds
