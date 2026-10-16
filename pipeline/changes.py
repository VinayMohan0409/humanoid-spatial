"""Helpers for drawing distances and regions: a distinct colour per region id, nearest-neighbour distances between two clouds, and a blue-to-red distance heatmap."""

import colorsys

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from config import HEATMAP_MAX


def region_colors(labels):
    """Return an RGB array giving each region id its own hue (36 hues, 70 degrees apart for consecutive ids) with alternating brightness, and light grey to unlabelled points."""
    colors = np.full((len(labels), 3), 0.85)
    for k in np.unique(labels[labels > 0]):
        colors[labels == k] = colorsys.hsv_to_rgb((k * 7 % 36) / 36, 1.0, (1.0, 0.65)[k % 2])
    return colors


def compute_cloud_distances(source, target):
    """Return, for every source point, the distance to its nearest target point."""
    tree = KDTree(np.asarray(target.points))
    distances, _ = tree.query(np.asarray(source.points))
    print(f"Computed distances for {len(distances)} points")
    return distances


def create_distance_heatmap(source, distances, max_dist=HEATMAP_MAX):
    """Return a copy of source coloured by distance (blue close, red far), with distances clipped at max_dist."""
    heatmap_pcd = o3d.geometry.PointCloud()
    heatmap_pcd.points = o3d.utility.Vector3dVector(np.asarray(source.points))

    clipped = np.minimum(distances, max_dist)
    min_dist = np.min(clipped)
    top_dist = np.max(clipped)

    if top_dist > min_dist:
        normalized_dists = (clipped - min_dist) / (top_dist - min_dist)
    else:
        normalized_dists = np.ones_like(clipped) * 0.5

    colors = np.zeros((len(distances), 3))
    colors[:, 0] = normalized_dists
    colors[:, 2] = 1 - normalized_dists
    colors[:, 1] = np.where(normalized_dists < 0.5, normalized_dists * 2, (1 - normalized_dists) * 2)

    heatmap_pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"Heatmap color scale: Blue = {min_dist:.4f}, Red = {top_dist:.4f} units")
    return heatmap_pcd
