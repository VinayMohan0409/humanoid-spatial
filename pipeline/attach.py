"""Attachment stage: projects each point of a video's fused cloud into that video's frames, averages the DINOv3 patch features of the frames that see it, and saves per-point attributes plus a PCA-coloured cloud for viewing."""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from config import DINO_DIR, FEATURES_DIR, KEEP_FRACTIONS, VIDEOS, VISIBILITY_TOL
from features import PATCH_SIZE
from point_clouds import load_reconstruction
from preprocess import preprocess_point_cloud
from tsdf import cloud_path


def load_points(name, keep=KEEP_FRACTIONS[0]):
    """Load a video's fused cloud, preprocess it (voxel merge, normals, outlier removal) and return points, colours and normals."""
    cloud = preprocess_point_cloud(o3d.io.read_point_cloud(str(cloud_path(name, keep))))
    return np.asarray(cloud.points), np.asarray(cloud.colors), np.asarray(cloud.normals)


def project(points, extrinsic, intrinsic, height, width):
    """Project world points into a camera; return pixel columns, rows, camera depth and an inside-the-image mask."""
    cam = points @ extrinsic[:, :3].T + extrinsic[:, 3]
    z = cam[:, 2]
    in_front = z > 1e-6
    z_safe = np.where(in_front, z, 1.0)
    u = np.round(intrinsic[0, 0] * cam[:, 0] / z_safe + intrinsic[0, 2])
    v = np.round(intrinsic[1, 1] * cam[:, 1] / z_safe + intrinsic[1, 2])
    inside = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.clip(u, 0, width - 1).astype(int), np.clip(v, 0, height - 1).astype(int), z, inside


def attach_features(name):
    """Average, per point, the DINOv3 patch features of the frames that see it, and save the per-point attributes."""
    recon = load_reconstruction(name)
    patch_features = np.load(DINO_DIR / f"{name}.npy")
    points, colors, normals = load_points(name)
    n_frames, height, width = recon["depth"].shape[:3]

    count = np.zeros(len(points), dtype=np.int32)
    feature_sum = np.zeros((len(points), patch_features.shape[-1]), dtype=np.float32)
    conf_sum = np.zeros(len(points))
    view_dir_sum = np.zeros((len(points), 3))

    for i in range(n_frames):
        extrinsic = recon["extrinsic"][i]
        u, v, z, inside = project(points, extrinsic, recon["intrinsic"][i], height, width)
        depth_i = recon["depth"][i][..., 0][v, u]
        visible = inside & (np.abs(z - depth_i) / depth_i < VISIBILITY_TOL)

        patch = patch_features[i][v[visible] // PATCH_SIZE, u[visible] // PATCH_SIZE].astype(np.float32)
        feature_sum[visible] += patch / np.linalg.norm(patch, axis=1, keepdims=True)
        conf_sum[visible] += recon["depth_conf"][i][v[visible], u[visible]]
        count += visible

        centre = -extrinsic[:, :3].T @ extrinsic[:, 3]
        to_camera = centre - points[visible]
        view_dir_sum[visible] += to_camera / np.linalg.norm(to_camera, axis=1, keepdims=True)

    seen = count > 0
    mean_feature = feature_sum[seen] / count[seen, None]
    agreement = np.linalg.norm(mean_feature, axis=1)  # 1 means every view gave the same feature direction
    features = np.zeros_like(feature_sum)
    features[seen] = mean_feature / agreement[:, None]
    if seen.any() and not seen.all():  # a point that no frame saw consistently has no feature; give it its nearest seen point's, so no later stage meets a zero vector
        features[~seen] = features[seen][cKDTree(points[seen]).query(points[~seen])[1]]
    spread = np.zeros(len(points))
    spread[seen] = 1 - agreement

    flip = np.einsum("ij,ij->i", normals, view_dir_sum) < 0  # face the normals towards the cameras that saw the point
    normals = np.where(flip[:, None], -normals, normals)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        FEATURES_DIR / f"{name}.npz",
        points=points,
        colors=colors,
        normals=normals,
        features=features.astype(np.float16),
        n_views=count,
        mean_conf=np.where(seen, conf_sum / np.maximum(count, 1), 0.0),
        feature_spread=spread,
    )
    print(f"{name}: {len(points):,} points | seen by >=1 frame {seen.mean():.1%}, >=3 frames {(count >= 3).mean():.1%} "
          f"| median views {np.median(count[seen]):.0f} | mean feature spread {spread[seen].mean():.3f}")


def pca_colors(feature_sets):
    """Map each feature set to RGB using the top 3 PCA components fitted jointly, so colours are comparable across videos."""
    sample = np.concatenate([f[::5] for f in feature_sets]).astype(np.float32)
    mean = sample.mean(axis=0)
    basis = np.linalg.svd(sample - mean, full_matrices=False)[2][:3]
    projected = [(f.astype(np.float32) - mean) @ basis.T for f in feature_sets]
    everything = np.concatenate(projected)
    low, high = np.percentile(everything, 2, axis=0), np.percentile(everything, 98, axis=0)
    return [np.clip((p - low) / (high - low), 0, 1) for p in projected]


def export_pca_clouds():
    """Write output/features3d/<name>_pca.ply for every video, coloured by DINOv3 features (grey where no frame saw the point)."""
    data = {name: np.load(FEATURES_DIR / f"{name}.npz") for name in VIDEOS}
    seen = {name: data[name]["n_views"] > 0 for name in VIDEOS}
    colors = pca_colors([data[name]["features"][seen[name]] for name in VIDEOS])
    for name, color in zip(VIDEOS, colors):
        rgb = np.full((len(seen[name]), 3), 0.5)
        rgb[seen[name]] = color
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(data[name]["points"])
        cloud.colors = o3d.utility.Vector3dVector(rgb)
        o3d.io.write_point_cloud(str(FEATURES_DIR / f"{name}_pca.ply"), cloud)


def attach_all():
    """Attach features to the points of every video and export the PCA-coloured clouds."""
    for name in VIDEOS:
        attach_features(name)
    export_pca_clouds()
