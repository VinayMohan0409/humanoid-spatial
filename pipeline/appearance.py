"""Appearance change stage: compares every point's DINOv3 feature and Lab chroma with the nearest point of the other video, so flat or swapped objects that the geometric heatmap cannot see still show up."""

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from changes import create_distance_heatmap
from compare import aligned_points
from config import APPEARANCE_MAX, CHECK_FRAME, CHROMA_MAX, PRESENT_RADIUS
from point_clouds import make_cloud
from segment import render_check

NO_MATCH_COLOR = (0.25, 0.25, 0.25)  # no surface in the other video here: the geometric heatmap covers it


def lab_ab(colors):
    """Return the Lab a,b (chroma) channels of RGB colours in 0-1; lightness is dropped so brightness changes do not count."""
    return cv2.cvtColor(colors.astype(np.float32).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)[:, 1:]


def appearance_maps(video, other_points, other):
    """Return per point: whether the other video has surface within PRESENT_RADIUS, and the feature dissimilarity (1 - cosine) and Lab chroma distance to that nearest surface point."""
    dist, idx = KDTree(other_points).query(video["points"])
    dissimilarity = np.clip(1 - np.einsum("ij,ij->i", video["features"], other["features"][idx]), 0, None)
    return dist <= PRESENT_RADIUS, dissimilarity, np.linalg.norm(lab_ab(video["colors"]) - lab_ab(other["colors"][idx]), axis=1)


def appearance_cloud(video, matched, values, max_value):
    """Return the video's points coloured blue (same appearance) to red (max_value or more) where matched, dark grey elsewhere."""
    heat = create_distance_heatmap(make_cloud(video["points"]), np.where(matched, values, 0.0), max_dist=max_value)
    return make_cloud(video["points"], np.where(matched[:, None], np.asarray(heat.colors), NO_MATCH_COLOR))


def export_appearance(before, after, out_dir):
    """Write per-video feature and chroma appearance maps (own coordinates) and their check-frame images to out_dir."""
    after_in_before, before_in_after = aligned_points(before, after)
    videos = {"before": before, "after": after}
    maps = {"before": appearance_maps(before, after_in_before, after), "after": appearance_maps(after, before_in_after, before)}
    for kind, column, limit in (("feature", 1, APPEARANCE_MAX), ("chroma", 2, CHROMA_MAX)):
        files = {}
        for name, video in videos.items():
            files[name] = out_dir / f"{name}_appearance_{kind}.ply"
            o3d.io.write_point_cloud(str(files[name]), appearance_cloud(video, maps[name][0], maps[name][column], limit))
        render_check(files, CHECK_FRAME, out_dir / f"appearance_{kind}_frame{CHECK_FRAME}.jpg")
