"""
PA-BBDM Training Script.

Usage:
    python train.py                           # default settings
    python train.py --epochs 100 --batch_size 8
    python train.py --resume checkpoints/epoch_30.pth
"""

import os, sys, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Ensure repo root and BBDM are on path
_PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJ)
_BBDM = os.path.join(os.path.dirname(_PROJ), 'BBDM', 'BBDM-main', 'BBDM-main')
sys.path.insert(0, _BBDM)

from data_loader import create_dataloader
from pabbdm.pa_bbdm import PABBDM, get_model_config
from utils import save_single_image
from ema import EMA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataroot', type=str, default='../dataset')
    p.add_argument('--out_dir', type=str, default='./checkpoints')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--print_every', type=int, default=100)
    p.add_argument('--val_every', type=int, default=10)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--val_samples', type=int, default=4)
    return p.parse_args()


def main():
    opt = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print('PA-BBDM: Polarization-Aware Brownian Bridge Diffusion Model')

    # ── Data ──
    train_loader = create_dataloader(opt.dataroot, phase='train',
        batch_size=opt.batch_size, modulo=10, preprocess='none',
        num_workers=opt.num_workers, pin_memory=True)
    val_loader = create_dataloader(opt.dataroot, phase='val',
        batch_size=1, modulo=10, preprocess='none',
        num_workers=0, pin_memory=True)
    print(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}')

    # ── Model ──
    cfg = get_model_config()
    model = PABBDM(cfg).to(device)
    model.is_train = True
    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.2f}M params')

    # ── EMA ──
    ema = EMA(ema_decay=0.995)
    ema.register(model)

    # ── Optimizer ──
    optimizer = torch.optim.Adam(model.get_parameters(), lr=opt.lr,
                                  weight_decay=0.0, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3000,
        threshold=0.0001, cooldown=3000, min_lr=5e-7)

    # ── Resume ──
    start_epoch = 0
    global_step = 0
    if opt.resume and os.path.exists(opt.resume):
        ckpt = torch.load(opt.resume, map_location=device)
        model.load_state_dict(ckpt['model'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['step']
        if 'ema_shadow' in ckpt:
            ema.shadow = {k: v.to(device) for k, v in ckpt['ema_shadow'].items()}
        print(f'Resumed from epoch {ckpt["epoch"]}')

    # ── Output ──
    out_dir = Path(opt.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_log = out_dir / 'loss_log.txt'

    # ── Val samples ──
    val_samples = []
    val_iter = iter(val_loader)
    for _ in range(opt.val_samples):
        try:
            b = next(val_iter)
            val_samples.append({'A': b['A'].clone(), 'B': b['B'].clone()})
        except StopIteration:
            break

    loss_history = []

    for epoch in range(start_epoch, opt.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{opt.epochs}')
        for batch in pbar:
            A = batch['A'].to(device)   # (B, 16, 256, 256)
            B = batch['B'].to(device)   # (B, 3, 256, 256)

            optimizer.zero_grad()
            loss, _ = model(B, A)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_parameters(), 1.0)
            optimizer.step()

            if global_step >= 30000 and global_step % 8 == 0:
                ema.update(model)

            epoch_loss += loss.item()
            epoch_batches += 1
            global_step += 1
            scheduler.step(loss.item())

            if epoch_batches % opt.print_every == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg': f'{epoch_loss/epoch_batches:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                })

        avg_loss = epoch_loss / max(epoch_batches, 1)
        elapsed = time.time() - epoch_start
        loss_history.append(avg_loss)

        with open(loss_log, 'a') as f:
            f.write(f'epoch {epoch+1}: avg_loss={avg_loss:.6f}  time={elapsed:.0f}s\n')
        print(f'Epoch {epoch+1}/{opt.epochs}  avg_loss={avg_loss:.6f}  '
              f'time={elapsed:.0f}s  lr={optimizer.param_groups[0]["lr"]:.2e}')

        # ── Save ──
        if (epoch + 1) % opt.save_every == 0 or epoch == 0:
            ckpt_path = out_dir / f'epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch, 'step': global_step,
                'model': model.state_dict(),
                'ema_shadow': ema.shadow,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss_history': loss_history,
            }, str(ckpt_path))
        torch.save({
            'epoch': epoch, 'step': global_step,
            'model': model.state_dict(),
            'ema_shadow': ema.shadow,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss_history': loss_history,
        }, str(out_dir / 'latest.pth'))

        # ── Val visualization ──
        if (epoch + 1) % opt.val_every == 0:
            model.eval()
            ema.apply_shadow(model)
            viz_dir = out_dir / 'viz' / f'epoch_{epoch+1:03d}'
            viz_dir.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                for i, s in enumerate(val_samples):
                    A = s['A'].to(device)
                    B = s['B']
                    fake = model.sample(A, clip_denoised=True)
                    save_single_image(A[0], str(viz_dir), f'sample{i}_input.png',
                                      to_normal=True)
                    save_single_image(fake[0].clamp(-1, 1), str(viz_dir),
                                      f'sample{i}_fake.png', to_normal=True)
                    save_single_image(B[0], str(viz_dir), f'sample{i}_real.png',
                                      to_normal=True)
            ema.restore(model)
            model.train()
            print(f'  Viz saved to {viz_dir}')

    print(f'\nTraining done. Loss: {[f"{l:.4f}" for l in loss_history]}')


if __name__ == '__main__':
    main()
