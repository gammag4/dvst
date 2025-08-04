import torch

from .encoder import DVSTEncoder


def regular_latent_aggregator(self: DVSTEncoder, latent_embeds, next_frames_embeds):
    # latent_embeds: (1, n_lat, d_model), next_frame_embeds
    # Just concatenates, gets result from transformer, and returns latents

    current_latent_embeds = latent_embeds
    for next_frame_embeds in next_frames_embeds:
        next_frame_embeds = next_frame_embeds.unsqueeze(0)
        
        embeds = torch.concat([next_frame_embeds, current_latent_embeds], dim=-2) # Concats embeddings with frame embeddings
        embeds = self.transformer(embeds) # Creates new embeds using tranformer
        current_latent_embeds = embeds[..., -latent_embeds.shape[-2]:, :] # Discards embeddings mapped from frame embeddings

    return current_latent_embeds


def residual_latent_aggregator(self: DVSTEncoder, latent_embeds, next_frames_embeds):
    # Concatenates, gets residuals from transformer, adds residuals to previous latents, and returns them

    current_latent_embeds = latent_embeds
    for next_frame_embeds in next_frames_embeds:
        next_frame_embeds = next_frame_embeds.unsqueeze(0)
        
        embeds = torch.concat([next_frame_embeds, current_latent_embeds], dim=-2) # Concats embeddings with frame embeddings
        embeds = self.transformer(embeds) # Creates new embeds using tranformer
        embeds = embeds[..., -latent_embeds.shape[-2]:, :] # Discards embeddings mapped from frame embeddings
        current_latent_embeds = embeds + current_latent_embeds # Adds residual

    return current_latent_embeds
