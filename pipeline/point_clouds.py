"""Shared loader for saved 3D reconstructions."""

import numpy as np

from config import RECON_DIR


def load_reconstruction(name):
    """Load a video's saved reconstruction arrays from output/vggt/<name>.npz."""
    return dict(np.load(RECON_DIR / f"{name}.npz"))
