"""Helpers shared by the change stages: loading a video's points, features and segment labels, moving points between the two videos' coordinates, the colours used to draw verdicts, and the see-through test (did the other video's cameras look past this spot, so nothing was there)."""

import numpy as np

from attach import project
from config import ALIGNMENT_DIR, FEATURES_DIR, KEEP_FRACTIONS, MIN_VIEW_FRACTION, SEE_THROUGH_MARGIN, SEGMENTS_DIR
from point_clouds import load_reconstruction
from tsdf import keep_mask

COLORS = {
    "background": (0.55, 0.55, 0.55),
    "unchanged": (0.55, 0.55, 0.55),
    "unobserved": (0.30, 0.30, 0.30),
    "added": (0.55, 0.85, 0.55),
    "removed": (0.95, 0.55, 0.55),
    "moved": (0.55, 0.65, 0.95),
}


def load_video(name):
    """Load a video's points, colours, features, segment labels and reconstruction."""
    data = np.load(FEATURES_DIR / f"{name}.npz")
    return {
        "points": data["points"],
        "colors": data["colors"],
        "features": data["features"].astype(np.float32),
        "labels": np.load(SEGMENTS_DIR / f"{name}_labels.npy"),
        "recon": load_reconstruction(name),
    }


def see_through_status(points, other):
    """Return -1 where enough of the other video's cameras saw past each point (in the other video's coordinates, so nothing was there), else 0."""
    recon = other["recon"]
    n_frames, height, width = recon["depth"].shape[:3]
    usable = keep_mask(recon["depth_conf"], KEEP_FRACTIONS[0])
    see_through = np.zeros(len(points), dtype=int)
    blocked = np.zeros(len(points), dtype=int)
    for i in range(n_frames):
        u, v, z, inside = project(points, recon["extrinsic"][i], recon["intrinsic"][i], height, width)
        ok = inside & usable[i][v, u]
        depth = recon["depth"][i][..., 0][v, u]
        see_through += ok & (depth > z + SEE_THROUGH_MARGIN)  # the camera saw past this spot, so nothing was there
        blocked += ok & (depth < z - SEE_THROUGH_MARGIN)  # something nearer hid this spot
    status = np.zeros(len(points), dtype=int)
    status[(see_through >= MIN_VIEW_FRACTION * n_frames) & (see_through > blocked)] = -1  # absent only if enough frames saw it empty
    return status


def aligned_points(before, after):
    """Return the after points in before's coordinates and the before points in after's own coordinates."""
    T = np.load(ALIGNMENT_DIR / "after_to_before.npy")
    return after["points"] @ T[:3, :3].T + T[:3, 3], (before["points"] - T[:3, 3]) @ T[:3, :3]
