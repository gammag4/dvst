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
        
    def create_scene_latents(self, videos, n_frames):
        return self.encoder(videos, n_frames)
    
    def get_frame(self, latent_embeds, Kinv, R, t, time, hw):
        return self.decoder(latent_embeds, Kinv, R, t, time, hw)
    
    def get_frames(self, latent_embeds, queries, n_frames):
        videos = []
        for v in queries:
            video = []
            for i in range(n_frames):
                Kinv = v.Kinv
                R = v.R if v.R.shape[0] == 1 else v.R[i:i+1]
                t = v.t if v.t.shape[0] == 1 else v.t[i:i+1]
                time = v.time[i:i+1]
                hw = v.shape[-2:]

                # TODO don't store all frames, compute grads after processing each frame to save memory (or batch of frames)
                video.append(self.get_frame(latent_embeds, Kinv, R, t, time, hw))
                
            videos.append(torch.concat(video))
        
        return videos
        
    # We assume videos are not big enough so that they need to be loaded in batches into memory #TODO load in batches if size exceed n_frames (create new scene for each batch of n_frames)
    def forward(self, scene):
        latent_embeds = self.create_scene_latents(scene.sources, scene.n_frames)
        I = self.get_frames(latent_embeds, scene.queries, scene.n_frames)
        return I
