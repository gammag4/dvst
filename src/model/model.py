import torch
import torch.nn as nn

from easydict import EasyDict as edict

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
        
        self.pose_encoder = PoseEncoder(self.config)
        self.encoder = DVSTEncoder(self.config, self.pose_encoder)
        self.decoder = DVSTDecoder(self.config, self.pose_encoder)
        
    def create_scene_latents(self, scene):
        return self.encoder(scene)
    
    def get_frame(self, latent_embeds, frame_query):
        return self.decoder(latent_embeds, frame_query)
    
    def get_frames(self, latent_embeds, frame_queries):
        frames = []
        for q in frame_queries:
            frames.append(self.get_frame(latent_embeds, q))
        
        return torch.concat(frames)
        
    # We assume videos are not big enough so that they need to be loaded in batches into memory #TODO load in batches if size exceed n_frames (create new scene for each batch of n_frames)
    def forward(self, scene):
        source_scene = edict(videos=scene.sources, n_frames=scene.n_frames)
        
        latent_embeds = self.create_scene_latents(source_scene)
        I = self.get_frames(latent_embeds, scene.queries)
        return I
