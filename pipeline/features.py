"""Feature extraction stage: runs DINOv3 on every reconstruction frame at VGGT's resolution and saves one patch-feature grid per frame."""

import numpy as np
import torch
from transformers import AutoModel

from config import DINO_DIR, DINO_MODEL, VIDEOS
from point_clouds import load_reconstruction
from reconstruction_3d import get_device

PATCH_SIZE = 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_model(device):
    """Load the DINOv3 model (from the local cache if present) onto the device."""
    return AutoModel.from_pretrained(DINO_MODEL).to(device).eval()


def frame_features(model, image, device):
    """Return the DINOv3 patch features, shaped (rows, cols, dim), of one uint8 CHW image."""
    x = (torch.from_numpy(image).float() / 255.0 - MEAN) / STD
    _, height, width = x.shape
    with torch.inference_mode():
        tokens = model(pixel_values=x[None].to(device)).last_hidden_state[0]
    patches = tokens[1 + model.config.num_register_tokens :]  # drop the class and register tokens
    return patches.reshape(height // PATCH_SIZE, width // PATCH_SIZE, -1).float().cpu().numpy()


def extract_video(model, name, device):
    """Extract features for every frame of a video and save them to output/dino/<name>.npy as float16."""
    recon = load_reconstruction(name)
    features = np.stack([frame_features(model, image, device) for image in recon["images"]]).astype(np.float16)
    DINO_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DINO_DIR / f"{name}.npy", features)
    print(f"{name}: features {features.shape} ({features.nbytes / 1e6:.0f} MB)")


def extract_all():
    """Extract DINOv3 features for every video in config.VIDEOS."""
    device = get_device()
    model = load_model(device)
    for name in VIDEOS:
        extract_video(model, name, device)
