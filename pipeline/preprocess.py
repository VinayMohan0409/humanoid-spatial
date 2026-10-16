"""Preprocessing: voxel-downsamples a fused cloud, estimates normals, removes statistical outliers and puts the points in a canonical order, so everything downstream gives the same result on every run."""

import numpy as np
import open3d as o3d

from config import OUTLIER_NEIGHBOURS, OUTLIER_STD_RATIO, VOXEL_SIZE

BEFORE_COLOR = (0.60, 0.85, 0.60)
AFTER_COLOR = (0.95, 0.60, 0.60)


def sort_points(cloud, voxel_size):
    """Return the cloud with its points sorted by position (rounded to a hundredth of a voxel). Open3D returns the same points of a fusion in a different order on every run, and the later stages depend on the order."""
    keys = np.round(np.asarray(cloud.points) / (voxel_size / 100)).astype(np.int64)
    return cloud.select_by_index(np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0])).tolist())


def preprocess_point_cloud(pcd, voxel_size=VOXEL_SIZE, remove_outliers=True):
    """Voxel-downsample, estimate normals, optionally remove statistical outliers, and sort the points into a canonical order."""
    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    if remove_outliers:
        down, _ = down.remove_statistical_outlier(nb_neighbors=OUTLIER_NEIGHBOURS, std_ratio=OUTLIER_STD_RATIO)
    return sort_points(down, voxel_size)
