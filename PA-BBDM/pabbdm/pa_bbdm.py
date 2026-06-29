"""
PA-BBDM: Polarization-Aware Brownian Bridge Diffusion Model.

Conditional diffusion model for virtual H&E staining from 16-channel
Mueller-matrix microscopy. A lightweight polarization encoder feeds
cross-attention context into the UNet at each denoising step.

Key differences from vanilla BBDM:
  - 16-channel condition (Mueller matrix) instead of 3-channel
  - SpatialTransformer + cross-attention at 32×32 bottleneck
  - SimplePolarizationEncoder: raw 16ch → 1×1 conv → pooled context
"""

import os, sys, numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ── BBDM base modules (from original BBDM codebase) ──
_PA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BBDM_ROOT = os.path.join(os.path.dirname(_PA_ROOT), 'BBDM', 'BBDM-main', 'BBDM-main')
if _BBDM_ROOT not in sys.path:
    sys.path.insert(0, _BBDM_ROOT)
from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel

from pabbdm.pa_encoder import PolarizationEncoder


def get_model_config():
    """Return default PA-BBDM configuration."""
    from argparse import Namespace
    cfg = Namespace()
    cfg.BB = Namespace()
    cfg.BB.params = Namespace()

    # Diffusion schedule
    cfg.BB.params.mt_type = 'linear'
    cfg.BB.params.objective = 'grad'
    cfg.BB.params.loss_type = 'l1'
    cfg.BB.params.skip_sample = True
    cfg.BB.params.sample_type = 'linear'
    cfg.BB.params.sample_step = 200
    cfg.BB.params.num_timesteps = 1000
    cfg.BB.params.eta = 1.0
    cfg.BB.params.max_var = 1.0

    # UNet
    cfg.BB.params.UNetParams = Namespace()
    cfg.BB.params.UNetParams.image_size = 256
    cfg.BB.params.UNetParams.in_channels = 19   # 16 (cond) + 3 (target)
    cfg.BB.params.UNetParams.model_channels = 64
    cfg.BB.params.UNetParams.out_channels = 3
    cfg.BB.params.UNetParams.num_res_blocks = 2
    cfg.BB.params.UNetParams.attention_resolutions = (32,)
    cfg.BB.params.UNetParams.channel_mult = (1, 2, 4, 8)
    cfg.BB.params.UNetParams.conv_resample = True
    cfg.BB.params.UNetParams.dims = 2
    cfg.BB.params.UNetParams.num_heads = 8
    cfg.BB.params.UNetParams.num_head_channels = 64
    cfg.BB.params.UNetParams.use_scale_shift_norm = True
    cfg.BB.params.UNetParams.resblock_updown = True
    cfg.BB.params.UNetParams.use_spatial_transformer = True
    cfg.BB.params.UNetParams.context_dim = 512
    cfg.BB.params.UNetParams.transformer_depth = 1
    cfg.BB.params.UNetParams.use_checkpoint = False
    cfg.BB.params.UNetParams.condition_key = 'SpatialRescaler'
    return cfg


class PABBDM(nn.Module):
    """
    Polarization-Aware Brownian Bridge Diffusion Model.

    Condition: 16-channel Mueller matrix → PolarizationEncoder → cross-attn context
    Target:    3-channel H&E → noise prediction via UNet
    """

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        mp = model_config.BB.params  # shorthand

        # ── Diffusion schedule ──
        self.num_timesteps = mp.num_timesteps
        self.mt_type = mp.mt_type
        self.max_var = getattr(mp, 'max_var', 1)
        self.eta = getattr(mp, 'eta', 1)
        self.skip_sample = mp.skip_sample
        self.sample_type = mp.sample_type
        self.sample_step = mp.sample_step
        self.steps = None
        self.register_schedule()

        self.loss_type = mp.loss_type
        self.objective = mp.objective
        self.image_size = mp.UNetParams.image_size
        self.channels = mp.UNetParams.in_channels
        self.condition_key = mp.UNetParams.condition_key

        # ── Denoising UNet with SpatialTransformer ──
        self.denoise_fn = UNetModel(**vars(mp.UNetParams))

        # ── Condition projection ──
        self.out_channels = mp.UNetParams.out_channels
        cond_in = mp.UNetParams.in_channels - self.out_channels
        self.cond_proj = nn.Conv2d(cond_in, self.out_channels, kernel_size=1)

        # ── Polarization context encoder ──
        ctx_dim = getattr(mp.UNetParams, 'context_dim', None) or 512
        self.polar_encoder = PolarizationEncoder(
            in_channels=cond_in, context_dim=ctx_dim, target_res=32)

    # ═══════════════════════════════════════════
    # Schedule
    # ═══════════════════════════════════════════

    def register_schedule(self):
        T = self.num_timesteps
        m_t = np.linspace(0.001, 0.999, T)
        m_tminus = np.append(0, m_t[:-1])
        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t
        to_torch = lambda x: torch.tensor(x, dtype=torch.float32)
        for name, val in [('m_t', m_t), ('m_tminus', m_tminus),
                          ('variance_t', variance_t),
                          ('variance_tminus', variance_tminus),
                          ('variance_t_tminus', variance_t_tminus),
                          ('posterior_variance_t', posterior_variance_t)]:
            self.register_buffer(name, to_torch(val))
        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(T - 1, 1,
                    step=-((T - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.tensor([1, 0]).long()), dim=0)

    def get_parameters(self):
        """Parameters for the optimizer."""
        params = list(self.denoise_fn.parameters())
        params += list(self.cond_proj.parameters())
        params += list(self.polar_encoder.parameters())
        return params

    # ═══════════════════════════════════════════
    # Training forward pass
    # ═══════════════════════════════════════════

    def forward(self, x, y, context=None):
        b, c, h, w, device = *x.shape, x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        y_bb = self.cond_proj(y) if self.cond_proj is not None else y
        cross_ctx = self.polar_encoder(y)
        return self.p_losses(x, y_bb, y, t, cross_ctx)

    def p_losses(self, x0, y, concat_ctx, t, cross_ctx, noise=None):
        b = x0.shape[0]
        noise = noise if noise is not None else torch.randn_like(x0)
        x_t, objective = self.q_sample(x0, y, t, noise)
        objective_recon = self.denoise_fn(x_t, timesteps=t, context=concat_ctx,
                                           cross_ctx=cross_ctx)
        recloss = (objective - objective_recon).abs().mean()
        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        return recloss, {"loss": recloss, "x0_recon": x0_recon}

    def q_sample(self, x0, y, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(x0)
        m_t = self.m_t[t].to(x0.device).view(-1, 1, 1, 1)
        var_t = self.variance_t[t].to(x0.device).view(-1, 1, 1, 1)
        sigma_t = torch.sqrt(var_t)
        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * noise
        else:
            raise NotImplementedError()
        return ((1. - m_t) * x0 + m_t * y + sigma_t * noise, objective)

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            return x_t - objective_recon
        raise NotImplementedError

    # ═══════════════════════════════════════════
    # Sampling
    # ═══════════════════════════════════════════

    @torch.no_grad()
    def p_sample(self, x_t, y, concat_ctx, cross_ctx, i, clip_denoised=False):
        b, device = x_t.shape[0], x_t.device
        t = torch.full((b,), self.steps[i], device=device, dtype=torch.long)
        objective_recon = self.denoise_fn(x_t, timesteps=t, context=concat_ctx,
                                           cross_ctx=cross_ctx)
        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        if clip_denoised:
            x0_recon.clamp_(-1., 1.)
        if self.steps[i] == 0:
            return x0_recon, x0_recon
        n_t = torch.full((b,), self.steps[i + 1], device=device, dtype=torch.long)
        m_t = self.m_t[t].to(device).view(1, 1, 1, 1)
        m_nt = self.m_t[n_t].to(device).view(1, 1, 1, 1)
        var_t = self.variance_t[t].to(device).view(1, 1, 1, 1)
        var_nt = self.variance_t[n_t].to(device).view(1, 1, 1, 1)
        sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
        sigma_t = torch.sqrt(sigma2_t) * self.eta
        noise = torch.randn_like(x_t)
        x_tminus_mean = ((1. - m_nt) * x0_recon + m_nt * y +
            torch.sqrt((var_nt - sigma2_t) / var_t) *
            (x_t - (1. - m_t) * x0_recon - m_t * y))
        return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample_loop(self, y, clip_denoised=True, sample_mid_step=False):
        y_bb = self.cond_proj(y) if self.cond_proj is not None else y
        concat_ctx = y
        cross_ctx = self.polar_encoder(y)
        img = y_bb
        for i in tqdm(range(len(self.steps)), desc='sampling',
                       total=len(self.steps)):
            img, _ = self.p_sample(x_t=img, y=y_bb, concat_ctx=concat_ctx,
                                    cross_ctx=cross_ctx, i=i,
                                    clip_denoised=clip_denoised)
        return img

    @torch.no_grad()
    def sample(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        return self.p_sample_loop(y, clip_denoised, sample_mid_step)
