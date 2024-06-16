"""
Copied and modified from:
https://github.com/facebookresearch/DiT/blob/main/models.py.
"""
import math
import torch
from torch import nn
from einops import repeat
from typing import Optional
from diffusion_grasp.models.common import Mlp, Attention

# https://github.com/HenryWJL/DiT/blob/main/models.py#L19
def modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    """FiLM modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# https://github.com/HenryWJL/DiT/blob/main/models.py#L27
class TimestepEmbedder(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: Optional[int] = 256
    ) -> None:
        """
        Embed scalar timesteps into vector representations.
        
        Args:
            hidden_size: the dimension of timestep embeddings.
        
            frequency_embedding_size: the dimension of positional embeddings.
        """
        super().__init__()
        self.mlp = Mlp(
            in_features=frequency_embedding_size,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=nn.SiLU
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: Optional[int] = 10000
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t: a 1-D Tensor of B indices, one per batch element (B,).
                          
            dim: the dimension of the output.
            
            max_period: controls the minimum frequency of the embeddings.
            
        Returns:
            The positional embeddings (B, D).
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Optional[torch.Tensor]) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    
# https://github.com/HenryWJL/DiT/blob/main/models.py#L101
class DiTBlock(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: Optional[float] = 4.0
    ) -> None:
        """
        Diffusion transformer block with adaLN-Zero conditioning.
        
        Args:
            hidden_size: the dimension of hidden layers.
            
            num_heads: the number of attention heads.
            
            mlp_ratio: mlp_hidden_dim = mlp_ratio * hidden_size.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6
        )
        self.attn = Attention(
            query_dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True
        )
        self.norm2 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: features (B, N, D).
            
            c: conditioning (B, D).
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

# https://github.com/HenryWJL/DiT/blob/main/models.py#L125
class FinalLayer(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        out_channels: int
    ) -> None:
        """
        The final layer of DiT.
        
        Args:
            hidden_size: the dimension of features.
            
            out_channels: the dimension of the output. 
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6
        )
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )
        
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

# https://github.com/HenryWJL/DiT/blob/main/models.py#L145
class DiT(nn.Module):
    
    def __init__(
        self,
        input_size: int,
        num_inputs: int,
        hidden_size: int,
        num_heads: int,
        depth: int,
        mlp_ratio: Optional[float] = 4.0
    ) -> None:
        """
        Diffusion transformer.
        
        Args:
            input_size: the dimension of inputs.
            
            num_inputs: the length of inputs.
        
            hidden_size: the dimension of features.
            
            num_heads: the number of attention head.
            
            depth: the number of DiT blocks.
            
            mlp_ratio: mlp_hidden_dim = mlp_ratio * hidden_size.
        """
        super().__init__()
        self.timestep_embedder = TimestepEmbedder(hidden_size)
        self.pos_embed = nn.Parameter(torch.rand(num_inputs, input_size))
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            out_channels=input_size
        )
        self.init_weights()

    def init_weights(self):
        # Initialize transformer layers.
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Initialize position embedding.
        torch.nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        # Initialize timestep embedding MLP.
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)
        # Zero-out adaLN modulation layers in DiT blocks.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # Zero-out output layers.
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: the inputs (B, N, D).
        
            t: the timesteps (B,).
        
            c: the conditioning (B, D).
        """
        x = x + repeat(self.pos_embed, 'n d -> b n d', b=x.shape[0])
        t = self.timestep_embedder(t)  # (B, D)
        c = t + c                               
        for block in self.blocks:
            x = block(x, c)                      
        x = self.final_layer(x, c)                               
        return x


        
