"""
Polarization-Aware Encoder for cross-attention conditioning.

Projects 16-channel Mueller matrix data to a context tensor consumed by
the UNet's SpatialTransformer cross-attention layers at each denoising step.

Unlike the full V2 model, this encoder does NOT use ChannelSelfAttention —
the raw Mueller features are directly projected via a 1×1 convolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolarizationEncoder(nn.Module):
    """Minimal encoder: 16ch → context_dim, pooled to target_res."""

    def __init__(self, in_channels=16, context_dim=512, target_res=32):
        super().__init__()
        self.target_res = target_res
        self.to_context = nn.Conv2d(in_channels, context_dim, kernel_size=1)

    def forward(self, x_cond):
        ctx = self.to_context(x_cond)  # (B, context_dim, 256, 256)
        if ctx.shape[-1] != self.target_res:
            ctx = F.adaptive_avg_pool2d(ctx, (self.target_res, self.target_res))
        return ctx  # (B, context_dim, target_res, target_res)
