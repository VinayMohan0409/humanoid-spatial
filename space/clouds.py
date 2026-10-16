"""Read .ply point clouds into compact arrays and save or load them for the demo."""

import numpy as np

MAX_POINTS = 60000


def read_ply(ply_path):
    """Return float32 positions and uint8 colours of a .ply cloud, thinned to MAX_POINTS."""
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(ply_path))
    xyz, rgb = np.asarray(cloud.points, dtype=np.float32), (np.asarray(cloud.colors) * 255).round().astype(np.uint8)
    keep = np.random.default_rng(0).permutation(len(xyz))[:MAX_POINTS]
    return xyz[keep], rgb[keep]


def save_npz(xyz, rgb, path):
    """Store a cloud as a compressed .npz file."""
    np.savez_compressed(path, xyz=xyz, rgb=rgb)


def load_npz(path):
    """Return the positions and colours stored by save_npz."""
    data = np.load(path)
    return data["xyz"], data["rgb"]
