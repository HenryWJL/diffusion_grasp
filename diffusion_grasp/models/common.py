"""
Copied and modified from:
https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py
https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_io.py
"""
import torch
from torch import nn, einsum
from functools import partial
from einops import rearrange
from typing import Optional, Union


def default(val, d):
    """Set default values."""
    return val if val is not None else d

# https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py#L13
class Mlp(nn.Module):
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[nn.Module] = nn.GELU,
        norm_layer: Optional[nn.Module] = None,
        bias: Optional[bool] = True,
        dropout: Optional[float] = 0.,
        use_conv: Optional[bool] = False,
        ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = partial(nn.Conv1d, kernel_size=1) if use_conv else nn.Linear
        self.chunk = nn.Sequential(
            linear_layer(in_features, hidden_features, bias=bias),
            act_layer(),
            nn.Dropout(dropout),
            default(norm_layer, nn.Identity()),
            linear_layer(hidden_features, out_features, bias=bias),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) or (B, D, N)"""    
        return self.chunk(x)

# https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_io.py#L94
class Attention(nn.Module):
    
    def __init__(
        self,
        query_dim,
        context_dim: Union[int, None] = None,
        num_heads: Optional[int] = 8,
        qkv_bias: Optional[bool] = False
    ) -> None:
        super().__init__()
        assert query_dim % num_heads == 0
        head_dim = query_dim // num_heads
        context_dim = default(context_dim, query_dim)
        self.to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias),
        self.to_kv = nn.Linear(context_dim, query_dim * 2, bias=qkv_bias),    
        self.proj = nn.Linear(query_dim, query_dim)
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        
    def forward(
        self,
        query: torch.Tensor,
        context: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N, C).
            
            context: (B, N, D).
            
        Returns:
            (B, N, C).
        """
        h = self.num_heads
        q = self.to_q(query)
        context = default(context, query)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        attn = (einsum('b i d, b j d -> b i j', q, k) * self.scale).softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.proj(out)