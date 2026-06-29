"""
PA-BBDM Inference Script.

Usage:
    python inference.py --ckpt checkpoints/epoch_50.pth --input /path/to/sample/
    python inference.py --ckpt checkpoints/epoch_50.pth --input /path/ --output ./results/
"""

import os, sys, argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJ)
_BBDM = os.path.join(os.path.dirname(_PROJ), 'BBDM', 'BBDM-main', 'BBDM-main')
if _BBDM not in sys.path:
    sys.path.insert(0, _BBDM)
from pabbdm.pa_bbdm import PABBDM, get_model_config

CHANNEL_FILES = [
    "M11.png", "M12.png", "M13.png", "M14.png",
    "M21.png", "M22.png", "M23.png", "M24.png",
    "M31.png", "M32.png", "M33.png", "M34.png",
    "M41.png", "M42.png", "M43.png", "M44.png",
]


def load_condition(sample_dir):
    """Load 16-channel Mueller matrix from a directory of single-channel PNGs."""
    chs = []
    for ch in CHANNEL_FILES:
        path = os.path.join(sample_dir, ch)
        img = np.array(Image.open(path), dtype=np.float32)
        chs.append(img)
    tensor = torch.from_numpy(np.stack(chs, 0))
    tensor = tensor.float().div_(255).mul_(2).sub_(1)  # → [-1, 1]
    return tensor.unsqueeze(0)


def tensor_to_pil(tensor):
    """Convert [-1,1] tensor to PIL Image."""
    img = tensor.detach().cpu().float().numpy()
    img = (img + 1.0) / 2.0 * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    return Image.fromarray(img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Path to checkpoint')
    p.add_argument('--input', required=True, help='Sample directory with 16 PNGs')
    p.add_argument('--output', default='./output', help='Output directory')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)

    # Load model
    cfg = get_model_config()
    model = PABBDM(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)
    # Apply EMA if available
    if ckpt.get('ema_shadow') and ckpt['step'] >= 30000:
        for name, param in model.named_parameters():
            if name in ckpt['ema_shadow']:
                param.data = ckpt['ema_shadow'][name].to(device)
    model.eval()
    print(f'PA-BBDM loaded (epoch {ckpt["epoch"]+1}, step {ckpt["step"]})')

    # Load input
    A = load_condition(args.input).to(device)
    print(f'Input: {A.shape}')

    # Generate
    with torch.no_grad():
        fake = model.sample(A, clip_denoised=True)
    result = tensor_to_pil(fake[0].clamp(-1, 1))
    out_path = os.path.join(args.output, 'fake.png')
    result.save(out_path)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
