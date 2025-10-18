import torch
import torch.nn as nn
import xformers.ops as xops
import einx


# Bias is disabled in all layers following works that show better stability in large models with it disabled


class FF(nn.Module):
    def __init__(self, d_model: int, e_ff: float, dropout: float, act_layer: nn.Module):
        super().__init__()
        
        ff_dim = int(e_ff * d_model)
        
        self.model = nn.Sequential(
            nn.Linear(d_model, ff_dim, bias=False),
            act_layer(),
            nn.Linear(ff_dim, d_model, bias=False),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FMHA(nn.Module):
    # attn_bias is the bias added to QK^T matrix before softmax, and also serves as attention mask by setting -inf in values in all values to mask
    # attn_bias can either come from xformers.ops.fmha.attn_bias (faster) or be a tensor (slower)
    def __init__(self, dropout: float, attn_op: xops.AttentionOp):
        super().__init__()
        
        self.dropout = dropout
        self.attn_op = attn_op
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_bias: torch.Tensor | xops.AttentionBias | None = None):
        # Q: (B, Mq, H, Kqk), K: (B, Mkv, H, Kqk), V: (B, Mkv, H, Kv)
        # Returns: (B, Mq, H, Kv)
        # B batch size, M sequence length, H number of heads, K embedding size per head
        
        return xops.memory_efficient_attention(
            Q, K, V,
            attn_bias=attn_bias,
            p=self.dropout if self.training else 0.0,
            op=self.attn_op
        )


class SelfAttn(nn.Module):
    def __init__(self, d_model: int, d_attn: int, n_heads: int, use_qk_norm: bool, qk_norm_eps: float, dropout: float, attn_op: xops.AttentionOp):
        super().__init__()
        
        self.n_heads = n_heads
        
        assert d_attn % n_heads == 0, f'n_heads should divide d_attn'
        # assert d_inner % n_heads == 0, f'n_heads should divide d_inner'
        
        self.use_qk_norm = use_qk_norm
        d_head = d_attn // n_heads
        
        self.fmha = FMHA(dropout, attn_op)
        
        self.W_Q, self.W_K, self.W_V = [nn.Linear(d_model, d_attn, bias=False) for _ in range(3)]
        self.W_O = nn.Sequential(
            nn.Linear(d_attn, d_model, bias=False),
            nn.Dropout(dropout)
        )
        
        if self.use_qk_norm:
            self.q_norm, self.k_norm = [nn.RMSNorm(d_head, qk_norm_eps) for _ in range(2)]
    
    def forward(self, X: torch.Tensor, attn_bias: torch.Tensor | xops.AttentionBias | None = None):
        Q, K, V = self.W_Q(X), self.W_K(X), self.W_V(X)
        Q, K, V = [einx.rearrange('... (h k) -> ... h k', k, h=self.n_heads) for k in (Q, K, V)]
        
        # QK-Norm
        if self.use_qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)
        
        X = self.fmha(Q, K, V, attn_bias=attn_bias)
        X = einx.rearrange('... h k -> ... (h k)', X, h=self.n_heads)
        
        X = self.W_O(X)
        return X


class Block(nn.Module):
    def __init__(self, d_model: int, d_attn: int, n_heads: int, e_ff: float, use_qk_norm: bool, qk_norm_eps: float, dropout: float, act_layer: nn.Module, attn_op: xops.AttentionOp):
        super().__init__()
        
        # Using Pre-LN in both layers
        # Also using RMSNorm instead of LayerNorm
        
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = SelfAttn(d_model, d_attn, n_heads, use_qk_norm, qk_norm_eps, dropout, attn_op=attn_op)
        
        self.ff = nn.Sequential(
            nn.RMSNorm(d_model),
            FF(d_model, e_ff, dropout, act_layer)
        )
    
    def forward(self, X: torch.Tensor, attn_bias: torch.Tensor | xops.AttentionBias | None = None):
        X = self.attn(self.norm1(X), attn_bias=attn_bias) + X
        X = self.ff(X) + X
        return X


class Encoder(nn.Module):
    def __init__(self, n_blocks: int, d_model: int, d_attn: int, n_heads: int, e_ff: float, use_qk_norm: bool, qk_norm_eps: float, dropout: float, act_layer: nn.Module = nn.GELU, attn_op: xops.AttentionOp = xops.fmha.MemoryEfficientAttentionFlashAttentionOp):
        super().__init__()
        
        self.blocks = nn.ModuleList([Block(d_model, d_attn, n_heads, e_ff, use_qk_norm, qk_norm_eps, dropout, act_layer, attn_op=attn_op) for _ in range(n_blocks)])
    
    def forward(self, X: torch.Tensor, attn_bias: torch.Tensor | xops.AttentionBias | None = None):
        shape = X.shape
        X = einx.rearrange('... n d -> (...) n d', X)
        
        for block in self.blocks:
            X = block(X, attn_bias=attn_bias)
        
        X = X.reshape(shape)
        return X


# TODO check new torch flash attention and compare with xformers:
# https://docs.pytorch.org/docs/stable/backends.html#module-torch.backends.mha
# torch.functional.scaled_dot_product_attention
# torch.nn.MultiheadAttention
# torch.nn.TransformerEncoderLayer
