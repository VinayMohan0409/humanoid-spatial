"""Segmentation stage: peels the large planes (floor, table, walls) off each video's cloud with RANSAC, then grows regions over the remaining points with 3D nearest neighbours gated by DINOv3 cosine similarity, so each object gets its own colour."""

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from attach import project
from changes import detect_missing_regions, region_colors
from config import (
    CHECK_FRAME,
    FEATURES_DIR,
    FEATURE_SIM_THRESHOLD,
    MAX_PLANES,
    MIN_PLANE_FRACTION,
    OBJECT_MIN_POINTS,
    PLANE_BAND,
    PLANE_FEATURE_SIM,
    SEGMENTS_DIR,
    VIDEOS,
)
from point_clouds import load_reconstruction


def peel_planes(points, features):
    """Repeatedly fit the largest plane to the unassigned points and remove it; a point counts as on the plane if it is within PLANE_BAND of it and its feature resembles the plane's mean feature. Returns a plane id per point (0 = not on a plane)."""
    plane_id = np.zeros(len(points), dtype=int)
    for k in range(1, MAX_PLANES + 1):
        free = np.where(plane_id == 0)[0]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points[free])
        (a, b, c, d), _ = cloud.segment_plane(PLANE_BAND, 3, 3000)
        normal = np.array([a, b, c])
        near = free[np.abs(points[free] @ normal + d) / np.linalg.norm(normal) < PLANE_BAND]

        mean = features[near].mean(axis=0)
        mean /= np.linalg.norm(mean)
        on_plane = near[features[near] @ mean > PLANE_FEATURE_SIM]
        if len(on_plane) < MIN_PLANE_FRACTION * len(points):
            break
        plane_id[on_plane] = k
    return plane_id


def segment_video(name):
    """Remove the planes from one video's cloud (saved as <name>_no_planes.ply in its real colours), grow regions over the rest and save output/segments/<name>_segments.ply."""
    data = np.load(FEATURES_DIR / f"{name}.npz")
    points, features = data["points"], data["features"].astype(np.float32)

    plane_id = peel_planes(points, features)
    print(f"\n=== {name}: {len(points):,} points ===")
    print("planes removed (points):", [int((plane_id == k).sum()) for k in range(1, plane_id.max() + 1)])

    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    remaining = o3d.geometry.PointCloud()
    remaining.points = o3d.utility.Vector3dVector(points[plane_id == 0])
    remaining.colors = o3d.utility.Vector3dVector(data["colors"][plane_id == 0])
    o3d.io.write_point_cloud(str(SEGMENTS_DIR / f"{name}_no_planes.ply"), remaining)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    # points on a plane get distance 0, so the region growing only runs over the remaining points
    regions, _, labels = detect_missing_regions(
        cloud, np.where(plane_id == 0, np.inf, 0.0), distance_threshold=1.0,
        region_size_threshold=OBJECT_MIN_POINTS, features=features, similarity_threshold=FEATURE_SIM_THRESHOLD,
    )

    colors = region_colors(labels)
    colors[labels == 0] = 0.45
    colors[plane_id > 0] = 0.85
    cloud.colors = o3d.utility.Vector3dVector(colors)

    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(SEGMENTS_DIR / f"{name}_segments.ply"), cloud)
    np.save(SEGMENTS_DIR / f"{name}_labels.npy", labels)


def render_check(cloud_files, frame, out_path):
    """Save an image showing, for each video, its camera frame next to its coloured cloud (cloud_files[name]) projected into that frame."""
    rows = []
    for name in VIDEOS:
        recon = load_reconstruction(name)
        image = recon["images"][frame].transpose(1, 2, 0)[:, :, ::-1].copy()
        height, width = image.shape[:2]

        cloud = o3d.io.read_point_cloud(str(cloud_files[name]))
        points, colors = np.asarray(cloud.points), (np.asarray(cloud.colors)[:, ::-1] * 255).astype(np.uint8)
        u, v, z, inside = project(points, recon["extrinsic"][frame], recon["intrinsic"][frame], height, width)

        canvas = np.full((height, width, 3), 20, np.uint8)
        for i in np.argsort(-z):  # far points first, so near points overwrite them
            if inside[i]:
                canvas[v[i], u[i]] = colors[i]
        canvas = cv2.dilate(canvas, np.ones((3, 3), np.uint8))

        for panel, label in ((image, f"{name} frame {frame}"), (canvas, Path(cloud_files[name]).stem)):
            panel = panel.copy()
            cv2.putText(panel, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            rows.append(panel)
    cv2.imwrite(str(out_path), np.vstack([np.hstack(rows[:2]), np.hstack(rows[2:])]))


def save_check(frame=CHECK_FRAME):
    """Save output/segments/check_frame<N>.jpg: for each video, the camera frame next to its segmented points projected into that frame."""
    render_check({name: SEGMENTS_DIR / f"{name}_segments.ply" for name in VIDEOS}, frame, SEGMENTS_DIR / f"check_frame{frame}.jpg")


def segment_all():
    """Segment every video in config.VIDEOS separately and save the frame check image."""
    o3d.utility.random.seed(0)
    for name in VIDEOS:
        segment_video(name)
    save_check()
