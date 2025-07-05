import torch
import torch.nn as nn


class DVSTEncoder(nn.Module):
    # latent_aggregator(next_frame_embeds, current_latent_embeds) creates next latent embeds from current embeds and next frame
    def __init__(self, dvst_config, pose_encoder):
        super().__init__()
        
        self.dvst_config = dvst_config
        self.n_lat = self.dvst_config.n_lat
        self.d_lat = self.dvst_config.d_lat
        
        self.start_latent_embeds = nn.Parameter(torch.zeros((self.n_lat, self.d_lat)))
        self.pose_encoder = pose_encoder
        self.latent_aggregator = self.dvst_config.latent_aggregator
        self.transformer = lambda x: x #TODO
        
    def forward(self, scene):
        #TODO ADD RESIDUAL BLOCKS IN BETWEEN
        # For each frame time, gets all frames from all cams in that time, passes them through the transformer and then goes to the next frame time
        current_embeds = self.start_latent_embeds
        n_frames = scene['shape'][-4]
        for i in range(n_frames):
            for s in scene:
                Kinv, R, t, time, video = [s[i] for i in ('Kinv', 'R', 't', 'time', 'video')]
                I = video[i]
                
                frame_embeds, _ = self.pose_encoder(Kinv, R, t, time, I) # Computes frame embeddings
                current_embeds = self.latent_aggregator(frame_embeds, current_embeds)
                
        return current_embeds
