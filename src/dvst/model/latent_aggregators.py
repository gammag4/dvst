import torch

from .encoder import DVSTEncoder


def regular_latent_aggregator(self: DVSTEncoder, latent_embeds, next_frames_embeds):
    # latent_embeds: (1, n_lat, d_model), next_frame_embeds
    # Just concatenates, gets result from transformer, and returns latents

    # Normalizes to latent space distribution
    current_latent_embeds = self.latent_norm(latent_embeds)
    next_frames_embeds = self.latent_norm(next_frames_embeds)
    
    for next_frame_embeds in next_frames_embeds:
        next_frame_embeds = next_frame_embeds.unsqueeze(0)
        
        embeds = torch.concat([next_frame_embeds, current_latent_embeds], dim=-2) # Concats embeddings with frame embeddings
        embeds = self.transformer(embeds) # Creates new embeds using tranformer
        embeds = embeds[..., -latent_embeds.shape[-2]:, :] # Discards embeddings mapped from frame embeddings
        embeds = self.latent_norm(embeds) # Normalizes to latent space distribution
        current_latent_embeds = embeds

    return current_latent_embeds


def residual_latent_aggregator(self: DVSTEncoder, latent_embeds, next_frames_embeds):
    # Concatenates, gets residuals from transformer, adds residuals to previous latents, and returns them
    
    # Normalizes to latent space distribution
    # In residual, we never normalize current latent embeds to not break the residuals from start latent embeds
    current_latent_embeds = latent_embeds
    next_frames_embeds = self.latent_norm(next_frames_embeds)
    
    for next_frame_embeds in next_frames_embeds:
        next_frame_embeds = next_frame_embeds.unsqueeze(0)
        
        embeds = torch.concat([next_frame_embeds, current_latent_embeds], dim=-2) # Concats embeddings with frame embeddings
        embeds = self.transformer(embeds) # Creates new embeds using tranformer
        embeds = embeds[..., -latent_embeds.shape[-2]:, :] # Discards embeddings mapped from frame embeddings
        
        # Normalizes new embeds and adds residual
        current_latent_embeds = self.latent_norm(embeds) + current_latent_embeds

    return current_latent_embeds


# Normalizes latent space after adding residual
def residual_latent_aggregator_w_normalized_latent_embeds(self: DVSTEncoder, latent_embeds, next_frames_embeds):
    # Concatenates, gets residuals from transformer, adds residuals to previous latents, and returns them
    
    # Normalizes to latent space distribution
    current_latent_embeds = self.latent_norm(latent_embeds)
    next_frames_embeds = self.latent_norm(next_frames_embeds)
    
    for next_frame_embeds in next_frames_embeds:
        next_frame_embeds = next_frame_embeds.unsqueeze(0)
        
        embeds = torch.concat([next_frame_embeds, current_latent_embeds], dim=-2) # Concats embeddings with frame embeddings
        embeds = self.transformer(embeds) # Creates new embeds using tranformer
        embeds = embeds[..., -latent_embeds.shape[-2]:, :] # Discards embeddings mapped from frame embeddings
        
        # Adds residual and normalizes final embeds
        current_latent_embeds = self.latent_norm(embeds + current_latent_embeds)

    return current_latent_embeds

# Normalizes latent space in all computations
def residual_latent_aggregator_w_normalized_latent_embeds_all(self: DVSTEncoder, latent_embeds, next_frames_embeds):
    # Concatenates, gets residuals from transformer, adds residuals to previous latents, and returns them
    
    # Normalizes to latent space distribution
    current_latent_embeds = self.latent_norm(latent_embeds)
    next_frames_embeds = self.latent_norm(next_frames_embeds)
    
    for next_frame_embeds in next_frames_embeds:
        next_frame_embeds = next_frame_embeds.unsqueeze(0)
        
        embeds = torch.concat([next_frame_embeds, current_latent_embeds], dim=-2) # Concats embeddings with frame embeddings
        embeds = self.transformer(embeds) # Creates new embeds using tranformer
        embeds = embeds[..., -latent_embeds.shape[-2]:, :] # Discards embeddings mapped from frame embeddings
        
        # Normalizes new embeds, adds residual and normalizes final embeds
        current_latent_embeds = self.latent_norm(self.latent_norm(embeds) + current_latent_embeds)

    return current_latent_embeds
