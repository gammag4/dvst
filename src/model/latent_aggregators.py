import torch

from .encoder import DVSTEncoder


def regular_latent_aggregator_incremental(self: DVSTEncoder, next_frame_embeds, current_latent_embeds):
    # Just concatenates, gets result from transformer, and returns latents
    concat_embeds = torch.concat([next_frame_embeds, current_latent_embeds], dim=-2) # Concats embeddings with frame embeddings
    next_embeds = self.transformer(concat_embeds) # Creates new embeds using tranformer
    next_embeds = next_embeds[-current_latent_embeds.shape[-2]:] # Discards embeddings mapped from frame embeddings
    return next_embeds


def residual_latent_aggregator_incremental(self: DVSTEncoder, next_frame_embeds, current_latent_embeds):
    # Concatenates, gets residuals from transformer, adds residuals to previous latents, and returns them
    concat_embeds = torch.concat([next_frame_embeds, current_latent_embeds], dim=-2) # Concats embeddings with frame embeddings
    residual_embeds = self.transformer(concat_embeds) # Creates new embeds using tranformer
    residual_embeds = residual_embeds[..., -current_latent_embeds.shape[-2]:, :] # Discards embeddings mapped from frame embeddings
    next_embeds = residual_embeds + current_latent_embeds # Adds residual
    return next_embeds
