"""
Standalone dataloader for 16-channel microscopy input -> 3-channel RGB H&E output.

Directory structure:
    {data_root}/
        trainA/{sample_id}/    -- 16 single-channel PNGs (M11.png .. M44.png)
        trainB/{sample_id}.png -- 3-channel RGB target

Split: modulo-based, every Nth sample -> test, rest -> train.

Usage:
    from data_loader import create_dataloader

    train_loader = create_dataloader("/path/to/dataset", "train", batch_size=8)
    test_loader  = create_dataloader("/path/to/dataset", "test", batch_size=1)
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

DEFAULT_CHANNEL_ORDER = [
    "M11.png", "M12.png", "M13.png", "M14.png",
    "M21.png", "M22.png", "M23.png", "M24.png",
    "M31.png", "M32.png", "M33.png", "M34.png",
    "M41.png", "M42.png", "M43.png", "M44.png",
]


class PairedStainingDataset(Dataset):
    """Self-contained dataset for paired microscopy -> H&E virtual staining.

    Args:
        data_root:      path containing trainA/ and trainB/ subdirectories
        phase:          "train", "val", or "test"
        modulo:         every Nth sample goes to val/test (default 10, i.e. 10%)
        preprocess:     "none" | "resize_and_crop" | "crop" | "scale_width"
        load_size:      size to resize to (for resize/crop modes)
        crop_size:      final crop size (for crop modes)
        no_flip:        disable horizontal flip augmentation
        max_samples:    cap on number of samples (None = no limit)
        channel_files:  list of 16 filenames for input channels
    """

    def __init__(
        self,
        data_root,
        phase="train",
        modulo=10,
        preprocess="none",
        load_size=256,
        crop_size=256,
        no_flip=True,
        max_samples=None,
        channel_files=None,
    ):
        super().__init__()
        self.data_root = data_root
        self.phase = phase
        self.modulo = modulo
        self.preprocess = preprocess
        self.load_size = load_size
        self.crop_size = crop_size
        self.no_flip = no_flip
        self.channel_files = channel_files or DEFAULT_CHANNEL_ORDER

        # Scan trainA directories and trainB files
        trainA_root = os.path.join(data_root, "trainA")
        trainB_root = os.path.join(data_root, "trainB")

        a_dirs = set()
        for name in os.listdir(trainA_root):
            if os.path.isdir(os.path.join(trainA_root, name)):
                a_dirs.add(name)

        b_ids = set()
        for name in os.listdir(trainB_root):
            if name.endswith(".png"):
                b_ids.add(name.replace(".png", ""))

        all_ids = sorted(a_dirs & b_ids)

        # Modulo-based split
        if phase == "train":
            self.sample_ids = [x for i, x in enumerate(all_ids) if i % modulo != 0]
        elif phase in ("test", "val"):
            self.sample_ids = [x for i, x in enumerate(all_ids) if i % modulo == 0]
        else:
            self.sample_ids = all_ids

        if max_samples is not None and len(self.sample_ids) > max_samples:
            self.sample_ids = self.sample_ids[:max_samples]

        self.trainA_root = trainA_root
        self.trainB_root = trainB_root

        print(f"PairedStainingDataset [{phase}]: {len(self.sample_ids)} samples "
              f"(modulo={modulo})")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]

        # --- Load 16-channel input ---
        a_dir = os.path.join(self.trainA_root, sample_id)
        A_channels = []
        for fname in self.channel_files:
            img = Image.open(os.path.join(a_dir, fname))
            A_channels.append(np.array(img, dtype=np.uint8))
        A_np = np.stack(A_channels, axis=-1)  # (H, W, 16)

        # --- Load RGB target ---
        b_path = os.path.join(self.trainB_root, f"{sample_id}.png")
        B_img = Image.open(b_path).convert("RGB")

        # --- Spatial transforms (shared random params for A and B) ---
        ref_size = (A_np.shape[1], A_np.shape[0])  # (W, H)
        transform_params = self._get_params(ref_size)

        A = self._transform_multichannel(A_np, transform_params)
        B = self._transform_rgb(B_img, transform_params)

        return {"A": A, "B": B, "A_paths": a_dir, "B_paths": b_path}

    # ------------------------------------------------------------------
    #  Spatial transform helpers
    # ------------------------------------------------------------------

    def _get_params(self, size):
        """Return deterministic spatial transform params for a given (W, H)."""
        w, h = size
        crop_pos = (0, 0)
        flip = False

        if "crop" in self.preprocess:
            x = np.random.randint(0, max(0, w - self.crop_size) + 1)
            y = np.random.randint(0, max(0, h - self.crop_size) + 1)
            crop_pos = (x, y)

        if not self.no_flip:
            flip = np.random.random() > 0.5

        return {"crop_pos": crop_pos, "flip": flip}

    def _spatial_transform(self, pil_img, params):
        """Compose resize/crop/flip → ToTensor (no normalization)."""
        t_list = []

        if "resize" in self.preprocess:
            t_list.append(transforms.Resize(
                [self.load_size, self.load_size],
                transforms.InterpolationMode.BICUBIC,
            ))
        elif "scale_width" in self.preprocess:
            t_list.append(transforms.Lambda(
                lambda img: self._scale_width(img, self.load_size, self.crop_size)))

        if "crop" in self.preprocess:
            pos = params["crop_pos"]
            cs = self.crop_size
            t_list.append(transforms.Lambda(
                lambda img, p=pos, s=cs: self._crop(img, p, s)))

        if self.preprocess == "none":
            t_list.append(transforms.Lambda(
                lambda img: self._make_power_2(img, base=4)))

        if not self.no_flip and params["flip"]:
            t_list.append(transforms.Lambda(
                lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)))

        t_list.append(transforms.ToTensor())
        return transforms.Compose(t_list)(pil_img)

    def _transform_multichannel(self, A_np, params):
        """Apply spatial transforms per-channel, stack, then normalize to [-1, 1]."""
        processed = []
        for i in range(A_np.shape[-1]):
            ch_img = Image.fromarray(A_np[:, :, i], mode="L")
            processed.append(self._spatial_transform(ch_img, params))
        A = torch.cat(processed, dim=0)  # (C, H, W)
        return A * 2.0 - 1.0

    def _transform_rgb(self, B_img, params):
        """Apply spatial transforms to RGB image, normalize to [-1, 1]."""
        B = self._spatial_transform(B_img, params)
        return B * 2.0 - 1.0

    @staticmethod
    def _scale_width(img, target_size, crop_size):
        ow, oh = img.size
        if ow == target_size and oh >= crop_size:
            return img
        w = target_size
        h = int(max(target_size * oh / ow, crop_size))
        return img.resize((w, h), Image.BICUBIC)

    @staticmethod
    def _crop(img, pos, size):
        ow, oh = img.size
        x1, y1 = pos
        if ow > size or oh > size:
            return img.crop((x1, y1, x1 + size, y1 + size))
        return img

    @staticmethod
    def _make_power_2(img, base=4):
        ow, oh = img.size
        h = int(round(oh / base) * base)
        w = int(round(ow / base) * base)
        if h == oh and w == ow:
            return img
        return img.resize((w, h), Image.BICUBIC)


# ------------------------------------------------------------------
#  Factory
# ------------------------------------------------------------------

def create_dataloader(
    data_root,
    phase="train",
    batch_size=8,
    modulo=10,
    preprocess="none",
    load_size=256,
    crop_size=256,
    no_flip=None,
    max_samples=None,
    channel_files=None,
    num_workers=4,
    pin_memory=True,
):
    """Create a DataLoader for the paired staining dataset.

    Args:
        data_root:    path containing trainA/ and trainB/
        phase:        "train" | "val" | "test"
        batch_size:   samples per batch
        modulo:       every Nth sample → val/test (default 10)
        preprocess:   "none" | "resize_and_crop" | "crop" | "scale_width"
        load_size:    resize target size
        crop_size:    crop target size
        no_flip:      disable flip. Default: True for val/test, False for train
        max_samples:  cap dataset size (None = all)
        channel_files: list of 16 input channel filenames
        num_workers:  DataLoader workers
        pin_memory:   pin memory for faster GPU transfer

    Returns:
        torch.utils.data.DataLoader
    """
    if no_flip is None:
        no_flip = (phase != "train")

    dataset = PairedStainingDataset(
        data_root=data_root,
        phase=phase,
        modulo=modulo,
        preprocess=preprocess,
        load_size=load_size,
        crop_size=crop_size,
        no_flip=no_flip,
        max_samples=max_samples,
        channel_files=channel_files,
    )

    shuffle = (phase == "train")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(phase == "train"),
    )


# ------------------------------------------------------------------
#  Quick test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-fs/CLY/VirtualStaining/paired-staining/dataset"

    for ph in ("train", "test"):
        loader = create_dataloader(root, phase=ph, batch_size=4, modulo=10, num_workers=0)
        batch = next(iter(loader))
        print(f"[{ph}] A: {batch['A'].shape}, B: {batch['B'].shape}, "
              f"samples: {len(batch['A_paths'])}")
        print(f"      A range: [{batch['A'].min():.2f}, {batch['A'].max():.2f}]")
        print(f"      B range: [{batch['B'].min():.2f}, {batch['B'].max():.2f}]")
