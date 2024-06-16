"""
Copied and modified from:
https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_io.py
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional, Union, Tuple
from einops import repeat
from diffusion_grasp.models.common import Attention


def exists(val):
    return val is not None


class PreNorm(nn.Module):
    
    def __init__(
        self,
        query_dim: int,
        module: nn.Module,
        context_dim: Union[int, None] = None,
        use_adaLN: Optional[bool] = False,
        condition_dim: Union[int, None] = None,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(
            query_dim,
            elementwise_affine=False,
            eps=1e-6
        )
        self.norm_context = nn.LayerNorm(
            context_dim,
            elementwise_affine=False,
            eps=1e-6
        ) if exists(context_dim) else None
        adaLN_modulation = None
        if use_adaLN:
            assert exists(condition_dim)
            adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    condition_dim,
                    2 * query_dim
                )
            )
        self.module = module
        self.adaLN_modulation = adaLN_modulation
        
    def modulate(
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor
    ) -> torch.Tensor:
        """FiLM modulation."""
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        if exists(self.adaLN_modulation):
            c = kwargs['condition']
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = self.modulate(x, shift, scale)    
        return self.module(x, **kwargs)


class GEGLU(nn.Module):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 8),
            GEGLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class GraspVAE(nn.Module):
    
    def __init__(
        self,
        num_inputs: int,
        input_dim: int,
        num_latents: int,
        latent_dim: int,
        depth: int,
        cross_heads: Optional[int] = 1,
        latent_heads: Optional[int] = 8,
        dropout: Optional[float] = 0.2,
        condition_dim: Union[int, None] = None
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.queries = nn.Parameter(torch.randn(num_inputs, latent_dim))
        self.encoder = nn.ModuleDict({
            "cross_attn_ffn": nn.ModuleList([
                PreNorm(
                    query_dim=latent_dim,
                    module=Attention(
                        query_dim=latent_dim,
                        context_dim=input_dim,
                        num_heads=cross_heads
                    ),
                    context_dim=input_dim
                ),
                PreNorm(
                    query_dim=latent_dim,
                    module=FeedForward(latent_dim)
                )
            ]),
            "self_attn_ffn": nn.ModuleList(
                [
                    nn.ModuleList([
                        PreNorm(
                            query_dim=latent_dim,
                            module=Attention(
                                query_dim=latent_dim,
                                num_heads=latent_heads
                            ),
                            use_adaLN=True,
                            condition_dim=condition_dim
                        ),
                        PreNorm(
                            query_dim=latent_dim,
                            module=FeedForward(latent_dim),
                            use_adaLN=True,
                            condition_dim=condition_dim
                        )
                    ])
                ] 
                for _ in range(depth)
            ),
            "proj": nn.Linear(latent_dim, latent_dim * 2)
        })
        self.decoder = nn.ModuleDict({
            "cross_attn": PreNorm(
                query_dim=input_dim,
                module=Attention(
                    query_dim=input_dim,
                    context_dim=latent_dim,
                    num_heads=cross_heads
                ),
                context_dim=latent_dim
            ),
            "self_attn_ffn": nn.ModuleList(
                [
                    PreNorm(
                        query_dim=latent_dim,
                        module=Attention(
                            query_dim=latent_dim,
                            num_heads=latent_heads
                        ),
                        use_adaLN=True,
                        condition_dim=condition_dim
                    ),
                    PreNorm(
                        query_dim=latent_dim,
                        module=FeedForward(latent_dim),
                        use_adaLN=True,
                        condition_dim=condition_dim
                    )
                ] 
                for _ in range(depth)
            ) 
        })
    
    def reparametrize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
        ) -> torch.Tensor:
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return (mu + std * eps).squeeze()
    
    def encode(
        self,
        x: torch.Tensor,
        c: Union[torch.Tensor, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: the inputs (B, N, C).
            
            c: the conditionings (B, D).
        """
        z = repeat(self.latents, 'n d -> b n d', b=x.shape[0])
        # dropout
        if self.training and self.dropout > 0:
            z = F.dropout(z, self.dropout)
        # cross attention.
        cross_attn, ffn = self.encoder["cross_attn_ffn"]
        z = cross_attn(z, context=x) + z
        z = ffn(z) + z
        # self attention.
        for self_attn, ffn in self.encoder["self_attn_ffn"]:
            z = self_attn(z, condition=c) + z
            z = ffn(z, condition=c) + z
        mu, logvar = self.encoder["proj"](z).chunk(2, dim=-1)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar
    
    def decode(
        self,
        z: torch.Tensor,
        c: Union[torch.Tensor, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = repeat(self.queries, 'n d -> b n d', b=z.shape[0])
        # self attention.
        for self_attn, ffn in self.decoder["self_attn_ffn"]:
            z = self_attn(z, condtion=c) + z
            z = ffn(z, condition=c) + z
        # cross attention.
        y = self.encoder["cross_attn"](y, context=z) + y
        return y

    def compute_losses(
        self,
        x: torch.Tensor,
        c: Union[torch.Tensor, None] = None,
        kl_weights: Optional[float] = 10.0
    ) -> torch.Tensor:
        z, mu, logvar = self.encode(x, c)
        x_hat = self.decode(z)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).sum(-1).mean()
        mse_loss = F.mse_loss(x_hat, x)
        total_loss = mse_loss + kl_weights * kl_loss
        return total_loss