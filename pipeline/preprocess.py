"""Preprocessing stage: loads the TSDF point clouds, estimates normals and removes statistical outliers, then saves before in pale green and after in pale red so their alignment can be inspected."""

import open3d as o3d

from config import ALIGNMENT_DIR, KEEP_FRACTIONS, OUTLIER_NEIGHBOURS, OUTLIER_STD_RATIO, VOXEL_SIZE
from tsdf import cloud_path

BEFORE_COLOR = (0.60, 0.85, 0.60)
AFTER_COLOR = (0.95, 0.60, 0.60)


def preprocess_point_cloud(pcd, voxel_size=VOXEL_SIZE, remove_outliers=True):
    """Voxel-downsample, estimate normals and optionally remove statistical outliers."""
    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    if remove_outliers:
        down, _ = down.remove_statistical_outlier(nb_neighbors=OUTLIER_NEIGHBOURS, std_ratio=OUTLIER_STD_RATIO)
    return down


def export_alignment(keep=KEEP_FRACTIONS[0]):
    """Write the preprocessed before (pale green) and after (pale red) clouds to output/alignment/."""
    ALIGNMENT_DIR.mkdir(parents=True, exist_ok=True)
    for name, color in (("before", BEFORE_COLOR), ("after", AFTER_COLOR)):
        raw = o3d.io.read_point_cloud(str(cloud_path(name, keep)))
        clean = preprocess_point_cloud(raw)
        clean.paint_uniform_color(color)
        o3d.io.write_point_cloud(str(ALIGNMENT_DIR / f"{name}.ply"), clean)
        print(f"{name}: {len(raw.points):,} points -> {len(clean.points):,} after outlier removal")
