"""Segmentation stage: peels the large planes (floor, table, walls) off each video's cloud with RANSAC, then grows objects over the remaining points by radius, gated by the cosine similarity of their DINOv3 features to the object's running mean, so each object gets its own colour."""

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from attach import project
from changes import region_colors
from config import (
    CHECK_FRAME,
    FEATURES_DIR,
    MAX_PLANES,
    MIN_PLANE_FRACTION,
    OBJECT_GROW_SIM,
    OBJECT_LINK_SIM,
    OBJECT_MIN_POINTS,
    OBJECT_RADIUS,
    PLANE_BAND,
    PLANE_FEATURE_SIM,
    SEGMENTS_DIR,
    VIDEOS,
)
from point_clouds import load_reconstruction


RANSAC_ITERATIONS = 3000
RANSAC_SEED = 0


def fit_plane(points, band, rng):
    """Return (normal, d) of the plane n.x + d = 0 with the most points within band, the best of RANSAC_ITERATIONS random 3-point planes. Deterministic: NumPy's generator gives the same numbers on every machine, ties go to the first candidate, and nothing is multithreaded (Open3D's segment_plane is, so it varied between runs)."""
    sample = points[rng.integers(0, len(points), size=(RANSAC_ITERATIONS, 3))].astype(np.float64)
    normals = np.cross(sample[:, 1] - sample[:, 0], sample[:, 2] - sample[:, 0])
    length = np.linalg.norm(normals, axis=1)
    valid = length > 1e-12
    normals[valid] /= length[valid, None]
    normals[~valid] = 0
    offsets = -(normals * sample[:, 0]).sum(axis=1)
    counts = np.zeros(RANSAC_ITERATIONS, dtype=np.int64)
    for start in range(0, len(points), 20000):
        chunk = points[start:start + 20000].astype(np.float64)
        counts += (np.abs(chunk @ normals.T + offsets) <= band).sum(axis=0)
    counts[~valid] = -1
    best = int(np.argmax(counts))
    return normals[best], float(offsets[best])


def peel_planes(points, features):
    """Repeatedly fit the largest plane to the unassigned points and remove it; a point counts as on the plane if it is within PLANE_BAND of it and its feature resembles the plane's mean feature. Returns a plane id per point (0 = not on a plane)."""
    plane_id = np.zeros(len(points), dtype=int)
    rng = np.random.default_rng(RANSAC_SEED)
    for k in range(1, MAX_PLANES + 1):
        free = np.where(plane_id == 0)[0]
        normal, d = fit_plane(points[free], PLANE_BAND, rng)
        near = free[np.abs(points[free] @ normal + d) / np.linalg.norm(normal) < PLANE_BAND]

        mean = features[near].mean(axis=0)
        mean /= np.linalg.norm(mean)
        on_plane = near[features[near] @ mean > PLANE_FEATURE_SIM]
        if len(on_plane) < MIN_PLANE_FRACTION * len(points):
            break
        plane_id[on_plane] = k
    return plane_id


def grow_objects(points, features):
    """Grow each object over points within OBJECT_RADIUS of each other whose features have cosine above OBJECT_LINK_SIM, adding a neighbour only if its feature also has cosine above OBJECT_GROW_SIM to the object's running mean feature. Returns an object id per point (0 = none, 1..N largest first, objects under OBJECT_MIN_POINTS dropped)."""
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)
    pairs = cKDTree(points).query_pairs(OBJECT_RADIUS, output_type="ndarray")
    pairs = pairs[(features[pairs[:, 0]] * features[pairs[:, 1]]).sum(axis=1) > OBJECT_LINK_SIM]
    adj = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points),) * 2)
    adj = adj + adj.T
    group = np.full(len(points), -1)
    for seed in range(len(points)):
        if group[seed] != -1:
            continue
        group[seed] = seed
        total, frontier = features[seed].copy(), np.array([seed])
        while len(frontier):
            near = np.unique(adj[frontier].indices)
            near = near[group[near] == -1]
            mean = total / max(np.linalg.norm(total), 1e-9)
            frontier = near[features[near] @ mean > OBJECT_GROW_SIM]
            group[frontier] = seed
            total += features[frontier].sum(axis=0)
    ids, counts = np.unique(group, return_counts=True)
    largest_first = np.argsort(-counts)
    labels = np.zeros(len(points), dtype=int)
    for i, gid in enumerate(ids[largest_first][counts[largest_first] >= OBJECT_MIN_POINTS], 1):
        labels[group == gid] = i
    return labels


def segment_video(name):
    """Remove the planes from one video's cloud (saved as <name>_no_planes.ply in its real colours), grow objects over the rest and save output/segments/<name>_segments.ply."""
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
    labels = np.zeros(len(points), dtype=int)
    labels[plane_id == 0] = grow_objects(points[plane_id == 0], features[plane_id == 0])
    print("objects found:", labels.max())

    colors = region_colors(labels)
    colors[labels == 0] = 0.45
    colors[plane_id > 0] = 0.85
    cloud.colors = o3d.utility.Vector3dVector(colors)

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
    for name in VIDEOS:
        segment_video(name)
    save_check()
