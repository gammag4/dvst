class DVSTEncoder(nn.Module):
    def __init__(self, dvst_config, pose_encoder):
        super().__init__()
        
        self.dvst_config = dvst_config
        self.n_lat = self.dvst_config.n_lat
        self.d_lat = self.dvst_config.d_lat
        
        self.start_embeds = nn.Parameter(torch.zeros((self.n_lat, self.d_lat)))
        self.pose_encoder = pose_encoder
        self.transformer = lambda x: x #TODO

    def forward(self, scene):
        #TODO ADD RESIDUAL BLOCKS IN BETWEEN
        # For each frame time, gets all frames from all cams in that time, passes them through the transformer and then goes to the next frame time
        latent_embeds = self.start_embeds
        n_frames = scene['shape'][-4]
        for i in range(n_frames):
            for s in scene:
                Kinv, R, t, time, video = [s[i] for i in ('Kinv', 'R', 't', 'time', 'video')]
                I = video[i]
                
                frame_embeds, _ = self.pose_encoder(Kinv, R, t, time, I) # Computes frame embeddings
                new_embeds = torch.concat([frame_embeds, latent_embeds], dim=-2) # Concats embeddings with frame embeddings
                new_embeds = self.transformer(new_embeds) # Creates new embeds using tranformer
                latent_embeds = new_embeds[frame_embeds.shape[-2]:] # Discards embeddings mapped from frame embeddings
                
        return latent_embeds
