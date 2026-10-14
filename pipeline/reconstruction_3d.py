"""3D reconstruction stage: runs VGGT-Omega once on the before and after frames together, so both share one coordinate frame, and saves per-video cameras, depth, confidence and world points."""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

from config import CHECKPOINT_FILE, CHECKPOINT_REPO, FRAMES_DIR, IMAGE_RESOLUTION, RECON_DIR, VIDEOS


def get_device():
    """Return the best available torch device (cuda, then mps, then cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device):
    """Load VGGT-Omega with its cached checkpoint onto the device."""
    path = hf_hub_download(CHECKPOINT_REPO, CHECKPOINT_FILE, local_files_only=True)
    model = VGGTOmega(autocast=False).eval()  # the autocast path is CUDA-only
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device)


def list_frames():
    """Return all frame paths (VIDEOS order) and the number of frames per video."""
    paths, counts = [], {}
    for name in VIDEOS:
        files = sorted((FRAMES_DIR / name).glob("*.jpg"))
        paths += files
        counts[name] = len(files)
    return paths, counts


def unproject(depth_map, extrinsic, intrinsic):
    """Turn per-frame depth maps and cameras into world-space points (copied from the repo's demo_gradio.py)."""
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def reconstruct(paths, model, device):
    """Run one forward pass over all frames and return numpy arrays (one entry per frame)."""
    images = load_and_preprocess_images([str(p) for p in paths], image_resolution=IMAGE_RESOLUTION).to(device)
    with torch.inference_mode():
        pred = model(images)
    extrinsic, intrinsic = encoding_to_camera(pred["pose_enc"], images.shape[-2:])

    out = {
        "images": (images.float().cpu().numpy() * 255).round().astype(np.uint8),
        "depth": pred["depth"][0].float().cpu().numpy(),
        "depth_conf": pred["depth_conf"][0].float().cpu().numpy(),
        "extrinsic": extrinsic[0].float().cpu().numpy(),
        "intrinsic": intrinsic[0].float().cpu().numpy(),
    }
    out["world_points"] = unproject(out["depth"], out["extrinsic"], out["intrinsic"])
    return out


def save(preds, counts):
    """Split the joint reconstruction by video and save each as output/vggt/<name>.npz."""
    RECON_DIR.mkdir(parents=True, exist_ok=True)
    start = 0
    for name in VIDEOS:
        stop = start + counts[name]
        np.savez(RECON_DIR / f"{name}.npz", **{k: v[start:stop] for k, v in preds.items()})
        start = stop


def reconstruct_all():
    """Reconstruct every video's frames jointly and save the results."""
    paths, counts = list_frames()
    device = get_device()
    preds = reconstruct(paths, load_model(device), device)
    save(preds, counts)
    return counts
