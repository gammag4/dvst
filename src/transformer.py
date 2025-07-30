import einx

import torch.nn as nn
import xformers.ops as xops


# Bias is disabled in all layers following works that show better stability in large models with it disabled


class FF(nn.Module):
    def __init__(self, d_model, e_ff, act_layer, dropout):
        super().__init__()
        
        ff_dim = e_ff * d_model
        
        self.model = nn.Sequential(
            nn.Linear(d_model, ff_dim, bias=False),
            act_layer(),
            nn.Linear(ff_dim, d_model, bias=False),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.model(x)


class FMHA(nn.Module):
    # attn_bias is the bias added to QK^T matrix before softmax, and also serves as attention mask by setting -inf in values in all values to mask
    # attn_bias can either come from xformers.ops.fmha.attn_bias (faster) or be a tensor (slower)
    def __init__(self, dropout):
        super().__init__()
        
        self.dropout = dropout

    def forward(self, Q, K, V, attn_bias=None):
        # Q: (B, Mq, H, Kqk), K: (B, Mkv, H, Kqk), V: (B, Mkv, H, Kv)
        # Returns: (B, Mq, H, Kv)
        # B batch size, M sequence length, H number of heads, K embedding size per head

        return xops.memory_efficient_attention(
            Q, K, V,
            attn_bias=attn_bias,
            p=self.dropout if self.training else 0.0,
            op=xops.fmha.MemoryEfficientAttentionFlashAttentionOp
        )


class SelfAttn(nn.Module):
    def __init__(self, d_model, n_heads, use_qk_norm, qk_norm_eps, dropout):
        super().__init__()
        
        d_inner = d_model # TODO check with different inner_d_model later
        
        assert d_inner % n_heads == 0, f'n_heads should divide d_inner'

        self.use_qk_norm = use_qk_norm
        d_head = d_inner // n_heads

        self.fmha = FMHA(dropout)

        self.W_Q, self.W_K, self.W_V = [nn.Linear(d_model, d_inner, bias=False) for _ in range(3)]
        self.W_O = nn.Sequential(
            nn.Linear(d_inner, d_model, bias=False),
            nn.Dropout(dropout)
        )
        
        self.split_heads = lambda X: einx.rearrange('... (h k) -> ... h k', X, h=n_heads)
        self.join_heads = lambda X: einx.rearrange('... h k -> ... (h k)', X, h=n_heads)
        
        if self.use_qk_norm:
            self.q_norm, self.k_norm = [nn.RMSNorm(d_head, qk_norm_eps) for _ in range(2)]

    def forward(self, X, attn_bias=None):
        Q = self.split_heads(self.W_Q(X))
        K = self.split_heads(self.W_K(X))
        V = self.split_heads(self.W_V(X))

        # QK-Norm
        if self.use_qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        X = self.fmha(Q, K, V, attn_bias=attn_bias)
        X = self.join_heads(X)

        X = self.W_O(X)
        return X


class Block(nn.Module):
    def __init__(self, d_model, n_heads, e_ff, use_qk_norm, qk_norm_eps, act_layer, dropout):
        super().__init__()

        # Using Pre-LN in both layers
        # Also using RMSNorm instead of LayerNorm

        self.norm1 = nn.RMSNorm(d_model)
        self.attn = SelfAttn(d_model, n_heads, use_qk_norm, qk_norm_eps, dropout)

        self.ff = nn.Sequential(
            nn.RMSNorm(d_model),
            FF(d_model, e_ff, act_layer, dropout)
        )

    def forward(self, X, attn_bias=None):
        X = self.attn(self.norm1(X), attn_bias=attn_bias) + X
        X = self.ff(X) + X
        return X


class Encoder(nn.Module):
    def __init__(self, n_blocks, d_model, n_heads, e_ff, use_qk_norm, qk_norm_eps, act_layer, dropout):
        super().__init__()

        self.blocks = nn.ModuleList([Block(d_model, n_heads, e_ff, use_qk_norm, qk_norm_eps, act_layer, dropout) for _ in range(n_blocks)])

    def forward(self, X, attn_bias=None):
        for block in self.blocks:
            X = block(X, attn_bias=attn_bias)
        return X


# TODO check new torch flash attention and compare with xformers:
# https://docs.pytorch.org/docs/stable/backends.html#module-torch.backends.mha
# torch.functional.scaled_dot_product_attention
# torch.nn.MultiheadAttention
# torch.nn.TransformerEncoderLayer
