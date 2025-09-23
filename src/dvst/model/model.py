import torch.nn as nn

from src.dvst.config import DVSTModelConfig
from src.dvst.datasets.scene_dataset import Scene, View
from .pose_encoder import PoseEncoder
from .encoder import DVSTEncoder
from .decoder import DVSTDecoder


# TODO change to RawDVST, create DVST that also has CNN to reduce dims and PoseWrapper to add a pose estimator to both
class DVST(nn.Module):
    # not specified: H, W, C, N_{context}
    # n_heads should divide d_model
    # p should divide H and W (padding, cropping and resizing)
    def __init__(self, config: DVSTModelConfig):
        super().__init__()
        
        self.config = config
        self.scene_batch_size = self.config.scene_batch_size
        
        self.pose_encoder = PoseEncoder(self.config)
        self.encoder = DVSTEncoder(self.config, self.pose_encoder)
        self.decoder = DVSTDecoder(self.config, self.pose_encoder)
        
        self.loss = config.train.loss
    
    @property
    def start_latent_embeds(self):
        return self.encoder.start_latent_embeds
    
    def create_scene_latents(self, scene: Scene):
        latent_embeds = None
        latent_embeds = self.encoder(scene, latent_embeds)
        return latent_embeds
    
    def generate_frames(self, latent_embeds, video_query: View):
        return self.decoder(latent_embeds, video_query)
    
    def forward(self, scene: Scene, latent_embeds=None):
        loss = 0.
        
        latent_embeds = self.encoder(scene, latent_embeds)
        
        # TODO fix so that the loss starts from the beginning of the scene (queries/targets)
        for query, target in zip(scene.queries, scene.targets):
            frames = self.generate_frames(latent_embeds, query)
            # TODO fix so that frames from starting batches receive less weights every time since they get repeated more
            l = self.loss(frames, target.view) / query.shape[0]
            loss = loss + l
        
        return loss, latent_embeds
