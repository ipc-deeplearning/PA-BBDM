# PA-BBDM: Polarization-Aware Brownian Bridge Diffusion Model

Official PyTorch implementation of **PA-BBDM**, a conditional diffusion model for
virtual H&E staining from 16-channel Mueller matrix microscopy.

## Overview

PA-BBDM extends the Brownian Bridge Diffusion Model (BBDM) with a lightweight
**Polarization-Aware Encoder** that projects the 16-channel Mueller matrix into
a cross-attention context consumed by the UNet at each denoising step.

### Architecture

```
  16-ch Mueller  ──→  PolarizationEncoder  ──→  cross-attn context (512-d, 32×32)
       │                                                    │
       └──→  cond_proj (16→3)  ──→  concat with x_t  ──→  UNet  ──→  ε_θ
                                                              ↑
                                              timestep embedding
```

- **PolarizationEncoder**: 1×1 Conv2d(16, 512) + AdaptiveAvgPool2d → 32×32
- **UNet**: Standard BBDM UNet with SpatialTransformer at 32×32 resolution
- **Cross-Attention**: 8-head attention, Q from UNet features, K,V from polarization context
- **Conditioning**: Raw 16ch Mueller matrix concatenated with noisy target (same as vanilla BBDM)

## Installation

```bash
pip install torch torchvision numpy tqdm pillow scikit-image lpips clean-fid cellpose
```

## Data Preparation

Dataset structure:
```
dataset/
  trainA/{sample_id}/    # 16 single-channel PNGs (M11.png ~ M44.png)
  trainB/{sample_id}.png # 3-channel RGB H&E target
```

Train/val split is modulo-10 (every 10th sample → val).

## Training

```bash
python train.py \
    --dataroot ../dataset \
    --out_dir ./checkpoints \
    --batch_size 4 \
    --lr 1e-4 \
    --epochs 50 \
    --save_every 10 \
    --val_every 10
```

To resume:
```bash
python train.py --resume checkpoints/epoch_30.pth
```

## Inference

```bash
python inference.py \
    --ckpt checkpoints/epoch_50.pth \
    --input /path/to/sample_dir/ \
    --output ./results/
```

The input directory should contain 16 single-channel PNG files (`M11.png` ~ `M44.png`).

## Pretrained Weights

| Model | Epoch | SSIM | LPIPS | FID | Download |
|-------|-------|------|-------|-----|----------|
| PA-BBDM | 50 | 0.608 | 0.164 | 29.98 | [link] |

## Results

PA-BBDM outperforms all baselines on paired reconstruction metrics:

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MAE ↓ | FID ↓ |
|-------|--------|--------|---------|-------|-------|
| pix2pix | 19.77 | 0.569 | 0.203 | 0.078 | 54.65 |
| Reg-GAN | 19.92 | 0.557 | 0.195 | 0.076 | 44.31 |
| BBDM | 20.08 | 0.544 | 0.187 | 0.075 | 34.04 |
| **PA-BBDM** | **20.76** | **0.608** | **0.164** | **0.069** | **29.98** |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Learning rate | 1×10⁻⁴ |
| Batch size | 4 |
| Diffusion steps | 1000 (train) / 200 (sample) |
| EMA decay | 0.995 |
| GPU | NVIDIA RTX 3090 24GB |
| Training epochs | 50 |
| Data split | 12,352 train / 1,373 val (modulo-10) |

## Citation

```bibtex
@article{pa-bbdm,
  title={PA-BBDM: Polarization-Aware Brownian Bridge Diffusion Model for Virtual H&E Staining},
  author={},
  journal={},
  year={2025}
}
```

## License

This project is released for academic research purposes.
