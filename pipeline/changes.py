"""Change detection stage: measures nearest-neighbour distances between the aligned after cloud and the before cloud in both directions, summarises points beyond a threshold, and writes distance heatmap clouds for viewing."""

import colorsys

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from config import (
    ALIGNMENT_DIR,
    CHANGE_THRESHOLD,
    CHANGES_DIR,
    FEATURE_SIM_THRESHOLD,
    FEATURES_DIR,
    HEATMAP_MAX,
    REGION_MIN_POINTS,
    REGION_NEIGHBOURS,
)

GOLDEN_RATIO = 0.618034


def region_colors(labels):
    """Return an RGB array giving every region id its own pale colour (golden-ratio hue steps) and light grey to unlabelled points."""
    colors = np.full((len(labels), 3), 0.85)
    for k in np.unique(labels[labels > 0]):
        colors[labels == k] = colorsys.hsv_to_rgb((k * GOLDEN_RATIO) % 1.0, 0.45, 0.95)
    return colors


def compute_cloud_distances(source, target):
    """Return, for every source point, the distance to its nearest target point."""
    tree = KDTree(np.asarray(target.points))
    distances, _ = tree.query(np.asarray(source.points))
    print(f"Computed distances for {len(distances)} points")
    return distances


def analyze_changes(distances, threshold=CHANGE_THRESHOLD):
    """Report the points whose nearest-neighbour distance exceeds the threshold; return their indices and summary stats."""
    change_indices = np.where(distances > threshold)[0]
    change_distances = distances[change_indices]

    if len(change_distances) > 0:
        mean_change = np.mean(change_distances)
        max_change = np.max(change_distances)
        share = len(change_indices) / len(distances)

        print(f"Detected {len(change_indices)} points with significant change")
        print(f"Mean change: {mean_change:.4f} units, Max change: {max_change:.4f} units")
        print(f"Share of points changed: {share * 100:.2f}%")

        return change_indices, {
            "mean_change": mean_change,
            "max_change": max_change,
            "change_percentage": share * 100,
        }
    print("No significant changes detected")
    return [], {"mean_change": 0, "max_change": 0, "change_percentage": 0}


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


def detect_missing_regions(
    source,
    distances,
    distance_threshold=CHANGE_THRESHOLD,
    region_size_threshold=REGION_MIN_POINTS,
    features=None,
    similarity_threshold=FEATURE_SIM_THRESHOLD,
):
    """Group source points whose distance to the other cloud exceeds the threshold into connected regions (region growing over their nearest neighbours), growing only into neighbours whose unit-norm feature has cosine similarity above similarity_threshold when features are given, and dropping regions smaller than region_size_threshold."""
    source_points = np.asarray(source.points)
    is_missing = distances > distance_threshold
    missing_indices = np.where(is_missing)[0]

    if len(missing_indices) == 0:
        print("No significant differences detected")
        return [], np.array([], dtype=int), np.zeros(len(source_points), dtype=int)

    row_of = np.full(len(source_points), -1)
    row_of[missing_indices] = np.arange(len(missing_indices))
    source_tree = KDTree(source_points)
    _, neighbours = source_tree.query(source_points[missing_indices], k=REGION_NEIGHBOURS)

    all_regions = []
    processed = np.zeros(len(source_points), dtype=bool)

    for idx in missing_indices:
        if processed[idx]:
            continue

        current_region = [idx]
        processed[idx] = True

        i = 0
        while i < len(current_region):
            current_idx = current_region[i]
            for neighbour_idx in neighbours[row_of[current_idx]][1:]:
                if not processed[neighbour_idx] and is_missing[neighbour_idx]:
                    if features is not None and features[current_idx] @ features[neighbour_idx] <= similarity_threshold:
                        continue
                    current_region.append(neighbour_idx)
                    processed[neighbour_idx] = True
            i += 1

        if len(current_region) >= region_size_threshold:
            all_regions.append(current_region)

    all_missing_indices = []
    region_labels = np.zeros(len(source_points), dtype=int)
    for region_idx, region in enumerate(all_regions, 1):
        all_missing_indices.extend(region)
        region_labels[region] = region_idx

    sizes = sorted((len(r) for r in all_regions), reverse=True)
    print(f"Detected {len(all_regions)} regions with total {len(all_missing_indices)} points "
          f"(largest: {sizes[:5]})")
    return all_regions, np.array(all_missing_indices), region_labels


def region_cloud(source, region_labels):
    """Return a copy of source with each region in its own pale colour and all other points light grey."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(source.points))
    cloud.colors = o3d.utility.Vector3dVector(region_colors(region_labels))
    return cloud


def export_changes():
    """Write added (after vs before) and removed (before vs after) distance heatmaps to output/changes/."""
    before = o3d.io.read_point_cloud(str(ALIGNMENT_DIR / "before.ply"))
    after = o3d.io.read_point_cloud(str(ALIGNMENT_DIR / "after_aligned.ply"))

    CHANGES_DIR.mkdir(parents=True, exist_ok=True)
    for label, source, target in (("added", after, before), ("removed", before, after)):
        video = "after" if label == "added" else "before"
        print(f"\n=== {label}: distance from each {video} point to the other cloud ===")
        distances = compute_cloud_distances(source, target)
        analyze_changes(distances)
        heatmap = create_distance_heatmap(source, distances)
        o3d.io.write_point_cloud(str(CHANGES_DIR / f"{label}_heatmap.ply"), heatmap)

        features = np.load(FEATURES_DIR / f"{video}.npz")["features"].astype(np.float32)
        assert len(features) == len(source.points), "feature rows must match the cloud's points"
        _, _, region_labels = detect_missing_regions(source, distances, features=features)
        o3d.io.write_point_cloud(str(CHANGES_DIR / f"{label}_regions.ply"), region_cloud(source, region_labels))
