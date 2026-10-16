"""Shared loader for saved 3D reconstructions."""

import numpy as np
import open3d as o3d

from config import RECON_DIR


def make_cloud(points, colors=None):
    """Return an Open3D point cloud of the points, optionally coloured."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def load_reconstruction(name):
    """Load a video's saved reconstruction arrays from output/vggt/<name>.npz."""
    return dict(np.load(RECON_DIR / f"{name}.npz"))
