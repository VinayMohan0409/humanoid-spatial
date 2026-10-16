"""TSDF fusion stage: integrates a video's per-frame colour and depth, after a confidence cutoff, into one TSDF voxel grid and exports the fused surface as a point cloud .ply."""

import numpy as np
import open3d as o3d

from config import (
    FAR_DEPTH_FACTOR,
    KEEP_FRACTIONS,
    RECON_DIR,
    SDF_TRUNC,
    TSDF_DIR,
    VIDEOS,
    VOXEL_SIZE,
    WEIGHT_THRESHOLD,
)
from point_clouds import load_reconstruction

TRUNC_MULTIPLIER = SDF_TRUNC / VOXEL_SIZE
DEVICE = o3d.core.Device("CPU:0")
DEPTH_SCALE = 1.0  # depth goes in as float32, so it needs no rescaling


def median_depth():
    """Return the median depth over both videos, the scale that also sets the voxel size."""
    return float(np.median(np.concatenate([np.load(RECON_DIR / f"{name}.npz")["depth"].ravel() for name in VIDEOS])))


def keep_mask(conf, keep):
    """Return a mask keeping the top `keep` fraction of confidence values across the whole video."""
    return conf >= np.percentile(conf, (1 - keep) * 100)


def cloud_path(name, keep):
    """Return the output path of a video's fused point cloud for the current keep-fraction and voxel size."""
    return TSDF_DIR / f"{name}_keep{int(keep * 100)}_vox{VOXEL_SIZE:g}.ply"


def make_volume():
    """Create an empty TSDF voxel block grid (tsdf, weight, colour) at the configured voxel size."""
    return o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=VOXEL_SIZE,
        block_resolution=16,
        block_count=20000,
        device=DEVICE,
    )


def to_4x4(extrinsic):
    """Pad a 3x4 camera-from-world matrix to 4x4."""
    matrix = np.eye(4)
    matrix[:3, :4] = extrinsic
    return matrix


def integrate_frame(volume, image, depth, mask, intrinsic, extrinsic, far_limit):
    """Integrate one frame into the volume, treating masked-out (low-confidence) depth and depth beyond far_limit as invalid."""
    depth = np.where(mask, depth[..., 0], 0.0).astype(np.float32)[..., None]
    color = (image.transpose(1, 2, 0) / 255.0).astype(np.float32)  # Open3D needs float colour to go with float depth

    depth_img = o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(depth)))
    color_img = o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(color)))
    K = o3d.core.Tensor(intrinsic.astype(np.float64))
    E = o3d.core.Tensor(to_4x4(extrinsic))

    coords = volume.compute_unique_block_coordinates(depth_img, K, E, DEPTH_SCALE, far_limit, TRUNC_MULTIPLIER)
    volume.integrate(coords, depth_img, color_img, K, K, E, DEPTH_SCALE, far_limit, TRUNC_MULTIPLIER)


def fuse_video(name, keep, far_limit):
    """Fuse all frames of a video at the given keep-fraction and save it to cloud_path(name, keep)."""
    recon = load_reconstruction(name)
    mask = keep_mask(recon["depth_conf"], keep)

    volume = make_volume()
    for i in range(len(mask)):
        integrate_frame(volume, recon["images"][i], recon["depth"][i], mask[i], recon["intrinsic"][i], recon["extrinsic"][i], far_limit)

    cloud = volume.extract_point_cloud(weight_threshold=WEIGHT_THRESHOLD).to_legacy()
    TSDF_DIR.mkdir(parents=True, exist_ok=True)
    path = cloud_path(name, keep)
    o3d.io.write_point_cloud(str(path), cloud)
    print(f"{path.name}: {len(cloud.points):,} points")


def fuse_all():
    """Run TSDF fusion for every video and keep-fraction, with one far-depth limit shared by both videos."""
    far_limit = FAR_DEPTH_FACTOR * median_depth()
    print(f"ignoring depth beyond {far_limit:.2f} ({FAR_DEPTH_FACTOR:g} x median depth)")
    for name in VIDEOS:
        for keep in KEEP_FRACTIONS:
            fuse_video(name, keep, far_limit)
