import torch.nn as nn

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
        
        self.pose_encoder = PoseEncoder(self.config.model)
        self.encoder = DVSTEncoder(self.config.model, self.pose_encoder)
        self.decoder = DVSTDecoder(self.config.model, self.pose_encoder)
        
    # We assume videos are not big enough so that they need to be loaded in batches into memory #TODO load in batches if size exceed n_frames (create new scene for each batch of n_frames)
    def forward(self, scene, Kinv, R, t, time, hw):
        latent_embeds = self.encoder(scene)
        I = self.decoder(latent_embeds, Kinv, R, t, time, hw)
